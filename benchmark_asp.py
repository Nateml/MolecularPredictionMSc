#!/usr/bin/env python3
"""
Benchmark for asp_bridge.solve().

Measures wall-clock time per solve call across a set of representative
molecular formulas. Run before and after optimizations to compare.

Usage:
    python benchmark_asp.py
    python benchmark_asp.py --runs 5
    python benchmark_asp.py --formulas small
    python benchmark_asp.py --timeout 30
"""
from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from asp_bridge import solve
from predicates.compiler import compile_asp_constraint
from predicates.interface import PredictionEntry, PredictionInterface

# ---------------------------------------------------------------------------
# Memory tracking
# ---------------------------------------------------------------------------

def _current_rss_kb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except OSError:
        pass
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KB, macOS returns bytes
    return rss if sys.platform != "darwin" else rss // 1024


class PeakMemoryMonitor:
    """Context manager that polls VmRSS every `interval` seconds and records the peak."""

    def __init__(self, interval: float = 0.01):
        self._interval = interval
        self._peak_kb = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "PeakMemoryMonitor":
        self._peak_kb = _current_rss_kb()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            rss = _current_rss_kb()
            if rss > self._peak_kb:
                self._peak_kb = rss

    def __exit__(self, *_) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self._peak_kb / 1024

# ---------------------------------------------------------------------------
# Test formulas
# ---------------------------------------------------------------------------

# Each entry: (label, atom_counts_dict, optional_pred_entries_dict)
# atom_counts keys match what compile_asp_constraint / solve expect.

# Solve time is driven by degree of unsaturation (DoU = (2C+2+N-H)/2),
# not atom count alone. C9H18 (DoU=1) ~0.4s; C8H10 (DoU=4) ~10s.

_SMALL = [
    # DoU=0-1, < 0.1s
    ("C2H6O  (ethanol)  DoU=0", dict(num_carbons=2, num_hydrogens=6,  num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
    ("C3H6O  (propanone) DoU=1", dict(num_carbons=3, num_hydrogens=6,  num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
    ("C3H8O  (propanol)  DoU=0", dict(num_carbons=3, num_hydrogens=8,  num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
]

_MEDIUM = [
    # DoU=1-2, 0.2-3s
    ("C5H10O (pentanone) DoU=1", dict(num_carbons=5, num_hydrogens=10, num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
    ("C7H14O (heptanone) DoU=1", dict(num_carbons=7, num_hydrogens=14, num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
    ("C7H8   (toluene)   DoU=4", dict(num_carbons=7, num_hydrogens=8,  num_oxygens=0, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
]

_LARGE = [
    # DoU=4, 5-15s
    ("C8H10  (xylene)    DoU=4", dict(num_carbons=8, num_hydrogens=10, num_oxygens=0, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
    ("C6H7N  (aniline)   DoU=4", dict(num_carbons=6, num_hydrogens=7,  num_oxygens=0, num_nitrogens=1, num_sulfurs=0, num_phosphorus=0), {}),
]

_XL = [
    # DoU=4-5, 30s+ — use --timeout to cap
    ("C8H8   (styrene)   DoU=5", dict(num_carbons=8, num_hydrogens=8,  num_oxygens=0, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
    ("C7H8O  (cresol)    DoU=4", dict(num_carbons=7, num_hydrogens=8,  num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0), {}),
]

_CONSTRAINED = [
    # C7H8O with carbonyl constraint — tests constraint pruning on a large formula
    (
        "C7H8O +carbonyl DoU=4",
        dict(num_carbons=7, num_hydrogens=8, num_oxygens=1, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0),
        {166: PredictionEntry(pred_set=[1])},
    ),
    # C8H10 with ring constraint — tests constraint pruning on large formula
    (
        "C8H10 +ring     DoU=4",
        dict(num_carbons=8, num_hydrogens=10, num_oxygens=0, num_nitrogens=0, num_sulfurs=0, num_phosphorus=0),
        {160: PredictionEntry(pred_set=[1])},
    ),
]

_FORMULA_SETS = {
    "small":      _SMALL,
    "medium":     _MEDIUM,
    "large":      _LARGE,
    "xl":         _XL,
    "constrained":_CONSTRAINED,
    "all":        _SMALL + _MEDIUM + _LARGE + _XL + _CONSTRAINED,
}

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def _build_asp_code(atom_counts: dict, pred_entries: dict) -> list[str]:
    entries = dict(pred_entries)
    entries.update(atom_counts)
    interface = PredictionInterface.model_validate(entries)
    asp_code, _ = compile_asp_constraint(interface)
    return asp_code


def run_benchmark(
    formulas: list,
    n_runs: int,
    timeout: int,
) -> list[dict]:
    results = []
    for label, atom_counts, pred_entries in formulas:
        asp_code = _build_asp_code(atom_counts, pred_entries)

        times: list[float] = []
        peak_mbs: list[float] = []
        n_candidates = 0
        error: str | None = None

        print(f"  {label} ... ", end="", flush=True)
        for run in range(n_runs):
            try:
                with PeakMemoryMonitor() as mem:
                    t0 = time.monotonic()
                    candidates, count = solve(
                        asp_code,
                        atom_counts,
                        output_list_of_smiles=True,
                        timeout=timeout,
                    )
                    elapsed = time.monotonic() - t0
                times.append(elapsed)
                peak_mbs.append(mem.peak_mb)
                n_candidates = count
                print(f"{elapsed:.2f}s/{mem.peak_mb:.0f}MB ", end="", flush=True)
            except Exception as exc:
                error = str(exc)
                print(f"ERROR({exc}) ", end="", flush=True)
                break

        print()

        if times:
            results.append({
                "label": label,
                "runs": len(times),
                "mean_s": statistics.mean(times),
                "min_s": min(times),
                "max_s": max(times),
                "stdev_s": statistics.stdev(times) if len(times) > 1 else 0.0,
                "peak_mb": max(peak_mbs),
                "n_candidates": n_candidates,
                "error": error,
            })
        else:
            results.append({
                "label": label,
                "runs": 0,
                "mean_s": float("nan"),
                "min_s": float("nan"),
                "max_s": float("nan"),
                "stdev_s": float("nan"),
                "peak_mb": float("nan"),
                "n_candidates": 0,
                "error": error,
            })

    return results


def print_table(results: list[dict]) -> None:
    col_label = max(len(r["label"]) for r in results)
    header = (
        f"{'Formula':<{col_label}}  {'mean(s)':>8}  {'min(s)':>7}  {'max(s)':>7}"
        f"  {'stdev':>6}  {'peak MB':>7}  {'cands':>6}  {'runs':>4}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in results:
        if r["error"] and r["runs"] == 0:
            print(f"  {r['label']:<{col_label}}  ERROR: {r['error']}")
            continue
        print(
            f"  {r['label']:<{col_label}}"
            f"  {r['mean_s']:>8.3f}"
            f"  {r['min_s']:>7.3f}"
            f"  {r['max_s']:>7.3f}"
            f"  {r['stdev_s']:>6.3f}"
            f"  {r['peak_mb']:>7.1f}"
            f"  {r['n_candidates']:>6}"
            f"  {r['runs']:>4}"
        )
    print(sep)

    valid = [r for r in results if r["runs"] > 0]
    if valid:
        total = sum(r["mean_s"] * r["runs"] for r in valid)
        print(f"\n  Total wall-clock (all runs): {total:.2f}s")
        print(f"  Mean per solve call:         {statistics.mean(r['mean_s'] for r in valid):.3f}s")
        print(f"  Peak memory (max across all): {max(r['peak_mb'] for r in valid):.1f} MB")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=3, help="Repetitions per formula (default: 3)")
    p.add_argument(
        "--formulas",
        choices=list(_FORMULA_SETS.keys()),
        default="all",
        help="Which formula set to benchmark (default: all). xl uses 30s+ formulas, set --timeout accordingly.",
    )
    p.add_argument("--timeout", type=int, default=60, help="Per-solve timeout in seconds (default: 60)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    formulas = _FORMULA_SETS[args.formulas]

    print(f"ASP solve benchmark")
    print(f"  Formula set : {args.formulas} ({len(formulas)} formulas)")
    print(f"  Runs each   : {args.runs}")
    print(f"  Timeout     : {args.timeout}s")
    print()

    results = run_benchmark(formulas, n_runs=args.runs, timeout=args.timeout)
    print_table(results)


if __name__ == "__main__":
    main()
