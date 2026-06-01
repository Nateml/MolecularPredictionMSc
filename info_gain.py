#!/usr/bin/env python3
"""
Feature screening via information gain — Algorithm 1 from MSc_Project_1.pdf.

For each candidate feature, measures the average log-reduction in ASP candidate
set size when the ground-truth feature value is encoded as a hard constraint.

    Score(F_k) = (1/N) * sum_i [ log|C(S_i)| - log|C(S_i, F_k=f_i)| ]

This is the oracle variant: uses ground-truth feature values (perfect prediction).

Usage:
    python info_gain.py --n-samples 200 --output results/info_gain.csv
    python info_gain.py --features fv --n-samples 100 --timeout 30
    python info_gain.py --features maccs --n-samples 500 --seed 99
    python info_gain.py --n-samples 200 --jobs 8   # parallel across samples

-------------------------------------------------------------------------------
HOW IT WORKS
-------------------------------------------------------------------------------

Background
----------
The ASP solver (smiles.lp + clingo) enumerates all structurally valid SMILES
strings consistent with a given molecular formula. The baseline candidate set
C(S_i) for sample i contains every molecule the solver can construct from the
atom counts of that sample's ground-truth molecule. Knowing a feature value
F_k = f (e.g. "has a carbonyl group") lets us add a hard constraint to the
ASP program, shrinking the candidate set to C(S_i, F_k=f_i). The log-ratio
  log|C(S_i)| - log|C(S_i, F_k=f_i)|
is the realised information gain of observing feature F_k on sample i. Averaged
over the dataset it estimates the conditional mutual information I(M ; F_k | S).
A high score means knowing this feature strongly narrows the solution space.

Feature families
----------------
Two families of features are evaluated, controlled by --features:

  MACCS keys (--features maccs):
    166 binary structural keys from RDKit. Each key has an ASP predicate
    defined in predicates/definitions.py. For sample i, the ground-truth value
    is record['maccs'][key_id] ∈ {0, 1}. Constraint:
      value=1 → ':- not predicate.'   (molecule must have this substructure)
      value=0 → ':- predicate.'       (molecule must not have this substructure)
    Keys marked 'unsupported' in definitions.py (e.g. isotopes, fragments) are
    excluded — they cause UNSAT in the ASP program and are uninformative.

  FV onehot features (--features fv):
    Binned structural features from data/feature_vector_mixed_schema.json.
    Each real-valued feature (e.g. alcohol count, ring count) is divided into
    bins by threshold edges. Each bin becomes a separate binary ASP predicate,
    e.g. alcohol_bin_0 (:- alcohol_count(0)), alcohol_bin_1 (:- alcohol_count(1)).
    For sample i and bin k of feature F:
      ground-truth bin = k  → pred_set=[1] → ':- not bin_k_pred.'
      ground-truth bin ≠ k  → pred_set=[0] → ':- bin_k_pred.'
    This means EVERY sample contributes a gain value for EVERY bin: samples
    in the bin contribute via enforcing presence; samples outside it contribute
    via enforcing absence. Binary features (2 bins) will produce identical scores
    for both bins since the constraints are logically equivalent.

Algorithm (per sample)
----------------------
1.  Atom counts (C/H/N/O/S/P) are read from structural_features['element_*'].
2.  Baseline solve: run ASP with only atom count constraints → count n_base.
    If n_base=0 (formula is UNSAT, e.g. due to parity) or solve times out,
    skip this sample entirely for all features.
3.  For each feature F_k:
      a. Compute pred_set from ground-truth (see above).
      b. Run constrained ASP solve → count n_f.
      c. If n_f=0 (constraint made the formula UNSAT — indicates an ASP/feature
         mismatch, not a real result), skip this (sample, feature) pair.
      d. If timeout, record as timeout and skip.
      e. Otherwise record gain = log(n_base) - log(n_f).

Parallelism
-----------
With --jobs > 1, samples are distributed across worker processes using
ProcessPoolExecutor. Each worker runs steps 1–3 for one sample and returns a
dict mapping feature_key → gain value (or 'timeout'/'zero'/'skip'). The main
process aggregates these dicts into per-feature statistics.

Output
------
CSV sorted by descending mean_info_gain with columns:
  feature_key            integer key in MACCS_DEFINITIONS / FEATURE_VECTOR_DEFINITIONS
  feature_label          human-readable name
  mean_info_gain         average log-reduction over valid samples
  std_info_gain          standard deviation of per-sample log-reductions
  n_valid                samples that produced a finite gain value
  n_timeout              samples that timed out during constrained solve
  n_skipped_baseline_zero  samples dropped because baseline was 0 or timed out
  n_skipped_constrained_zero  (sample, feature) pairs where constrained solve → 0
  n_skipped_pred_none    (sample, feature) pairs skipped (feature not in record)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from predicates.definitions import FEATURE_VECTOR_DEFINITIONS, MACCS_DEFINITIONS
from predicates.interface import PredictionInterface, PredictionEntry
from predicates.compiler import compile_asp_constraint
from asp.bridge import solve, is_asp_supported

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature catalogue building
# ---------------------------------------------------------------------------

def _build_feature_specs(schema: dict, feature_families: str) -> list[dict]:
    """
    Return a list of serializable feature spec dicts, one per (feature, bin).

    Each spec has:
        key (int): FV or MACCS definition key
        label (str): human-readable label
        family (str): 'maccs' or 'fv'
        maccs_idx (int|None): index into record['maccs'] for MACCS features
        sf_name (str|None): structural_features key name for FV features
        bin_idx (int|None): which bin is this feature testing (for FV)
        edges (list|None): bin edges (for FV)
    """
    specs = []

    if feature_families in ("maccs", "both"):
        for key, defn in MACCS_DEFINITIONS.items():
            if defn.get("unsupported"):
                continue
            specs.append({
                "key": key,
                "label": f"MACCS_{key}_{defn['predicate']}",
                "family": "maccs",
                "maccs_idx": key,
                "sf_name": None,
                "bin_idx": None,
                "edges": None,
            })

    if feature_families in ("fv", "both"):
        feat_schema = schema["feature_schema"]
        pred_to_key = {v["predicate"]: k for k, v in FEATURE_VECTOR_DEFINITIONS.items()}

        for feat_name, spec in sorted(feat_schema.items()):
            if spec["type"] != "onehot":
                continue
            edges = spec["edges"]
            for oname in spec["onehot_names"]:
                pred_name = oname.split(":")[0]
                if pred_name not in pred_to_key:
                    continue
                m = re.search(r"_bin_(\d+)$", pred_name)
                if not m:
                    continue
                bin_idx = int(m.group(1))
                fv_key = pred_to_key[pred_name]
                specs.append({
                    "key": fv_key,
                    "label": f"FV_{feat_name}_bin{bin_idx}_{fv_key}",
                    "family": "fv",
                    "maccs_idx": None,
                    "sf_name": feat_name,
                    "bin_idx": bin_idx,
                    "edges": edges,
                })

    return specs


def _pred_set_for_spec(spec: dict, record: dict) -> list | None:
    """Return the ASP prediction set for this feature on this record, or None to skip."""
    if spec["family"] == "maccs":
        val = record["maccs"][spec["maccs_idx"]]
        return [int(val)]

    # fv: binary indicator for whether this bin is the ground-truth bin
    sf = record.get("structural_features", {})
    sf_name = spec["sf_name"]
    if sf_name not in sf:
        return None
    actual_bin = _assign_bin(float(sf[sf_name]), spec["edges"])
    return [1] if actual_bin == spec["bin_idx"] else [0]


def _assign_bin(value: float, edges: list[float]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


# ---------------------------------------------------------------------------
# ASP helpers
# ---------------------------------------------------------------------------

def _atom_counts(sf: dict) -> dict:
    return {
        "num_carbons":    int(sf.get("element_C", 0)),
        "num_hydrogens":  int(sf.get("element_H", 0)),
        "num_nitrogens":  int(sf.get("element_N", 0)),
        "num_oxygens":    int(sf.get("element_O", 0)),
        "num_sulfurs":    int(sf.get("element_S", 0)),
        "num_phosphorus": int(sf.get("element_P", 0)),
    }


def _solve_count(atom_counts: dict, extra_key: int | None, pred_set: list | None, timeout: int) -> tuple[int, bool]:
    """Run ASP solve with optional extra constraint; return (count, timed_out)."""
    pred_dict: dict = {**atom_counts}
    if extra_key is not None:
        pred_dict[extra_key] = PredictionEntry(pred_set=pred_set)
    interface = PredictionInterface.model_validate(pred_dict)
    asp_code, ac = compile_asp_constraint(interface)
    _, n, timed_out = solve(asp_code, ac, timeout=timeout)
    return n, timed_out


# ---------------------------------------------------------------------------
# Per-sample worker (runs in subprocess for parallelism)
# ---------------------------------------------------------------------------

def _process_sample(args_tuple: tuple) -> dict | None:
    """
    Worker: compute baseline + all constrained counts for one record.

    Returns dict: {spec_key: gain_float | 'timeout' | 'zero' | 'skip'} plus
    metadata keys '_n_base' and '_label'.

    Returns None if baseline is 0 or timed out.
    """
    record, feature_specs, timeout, sys_path = args_tuple

    # Re-add sys.path in subprocess (not inherited from parent)
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)

    import math
    from predicates.interface import PredictionInterface, PredictionEntry
    from predicates.compiler import compile_asp_constraint
    from asp.bridge import solve

    sf = record["structural_features"]
    atom_counts = {
        "num_carbons":    int(sf.get("element_C", 0)),
        "num_hydrogens":  int(sf.get("element_H", 0)),
        "num_nitrogens":  int(sf.get("element_N", 0)),
        "num_oxygens":    int(sf.get("element_O", 0)),
        "num_sulfurs":    int(sf.get("element_S", 0)),
        "num_phosphorus": int(sf.get("element_P", 0)),
    }

    def _run(extra_key=None, pred_set=None):
        pred_dict: dict = {**atom_counts}
        if extra_key is not None:
            pred_dict[extra_key] = PredictionEntry(pred_set=pred_set)
        iface = PredictionInterface.model_validate(pred_dict)
        code, ac = compile_asp_constraint(iface)
        _, n, timed_out = solve(code, ac, timeout=timeout)
        return n, timed_out

    n_base, timed_out = _run()
    if timed_out or n_base == 0:
        return None

    def _assign_bin(value, edges):
        for i, edge in enumerate(edges):
            if value <= edge:
                return i
        return len(edges)

    result = {"_n_base": n_base}
    for spec in feature_specs:
        key = spec["key"]
        # Compute pred_set
        if spec["family"] == "maccs":
            pred_set = [int(record["maccs"][spec["maccs_idx"]])]
        else:
            sf_name = spec["sf_name"]
            if sf_name not in sf:
                result[key] = "skip"
                continue
            actual_bin = _assign_bin(float(sf[sf_name]), spec["edges"])
            pred_set = [1] if actual_bin == spec["bin_idx"] else [0]

        n_f, to = _run(key, pred_set)
        if to:
            result[key] = "timeout"
        elif n_f == 0:
            result[key] = "zero"
        else:
            result[key] = math.log(n_base) - math.log(n_f)

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate(sample_results: list[dict], feature_specs: list[dict]) -> list[dict]:
    """Aggregate per-sample dicts into per-feature stats."""
    from collections import defaultdict

    gains: dict[int, list[float]] = defaultdict(list)
    n_timeout: dict[int, int] = defaultdict(int)
    n_zero: dict[int, int] = defaultdict(int)
    n_skip: dict[int, int] = defaultdict(int)
    n_baseline_zero = 0  # samples discarded before reaching feature loop

    for res in sample_results:
        if res is None:
            n_baseline_zero += 1
            continue
        for spec in feature_specs:
            k = spec["key"]
            v = res.get(k)
            if v is None or v == "skip":
                n_skip[k] += 1
            elif v == "timeout":
                n_timeout[k] += 1
            elif v == "zero":
                n_zero[k] += 1
            else:
                gains[k].append(v)

    rows = []
    for spec in feature_specs:
        k = spec["key"]
        gs = gains[k]
        n = len(gs)
        mean = sum(gs) / n if n > 0 else float("nan")
        std = (
            math.sqrt(sum((g - mean) ** 2 for g in gs) / n)
            if n > 1 else float("nan")
        )
        rows.append({
            "feature_key": k,
            "feature_label": spec["label"],
            "mean_info_gain": mean,
            "std_info_gain": std,
            "n_valid": n,
            "n_timeout": n_timeout[k],
            "n_skipped_baseline_zero": n_baseline_zero,
            "n_skipped_constrained_zero": n_zero[k],
            "n_skipped_pred_none": n_skip[k],
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",   default="data/multitask_mlp_hardcoded.test.json")
    parser.add_argument("--schema",    default="data/feature_vector_mixed_schema.json")
    parser.add_argument("--n-samples", type=int, default=200,
                        help="ASP-supported records to sample (default: 200).")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--timeout",   type=int, default=60,
                        help="Per-solve timeout in seconds (default: 60).")
    parser.add_argument("--features",  choices=["maccs", "fv", "both"], default="both")
    parser.add_argument("--output",    default="results/info_gain.csv")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Stop reading dataset after this many records.")
    parser.add_argument("--max-heavy-atoms", type=int, default=None,
                        help="Skip molecules with more heavy atoms than this.")
    parser.add_argument("--jobs",      type=int, default=1,
                        help="Parallel workers for sample processing (default: 1).")
    args = parser.parse_args()

    random.seed(args.seed)

    # -- Load ----------------------------------------------------------------
    log.info("Loading dataset: %s", args.dataset)
    try:
        import ijson
        with open(args.dataset, "rb") as f:
            all_records = []
            for rec in ijson.items(f, "item"):
                all_records.append(rec)
                if args.max_records and len(all_records) >= args.max_records:
                    break
    except ImportError:
        with open(args.dataset) as f:
            all_records = json.load(f)
        if args.max_records:
            all_records = all_records[:args.max_records]
    log.info("Loaded %d records", len(all_records))

    with open(args.schema) as f:
        schema = json.load(f)

    # -- Filter --------------------------------------------------------------
    log.info("Filtering records…")
    supported = []
    for r in all_records:
        if args.max_heavy_atoms is not None:
            sf = r.get("structural_features", {})
            if sf.get("heavy_atom_count", 0) > args.max_heavy_atoms:
                continue
        ok, _ = is_asp_supported(r["smiles"])
        if ok:
            supported.append(r)
    log.info("%d / %d records pass filters", len(supported), len(all_records))

    if len(supported) < args.n_samples:
        log.warning("Only %d supported; using all.", len(supported))
        samples = supported
    else:
        samples = random.sample(supported, args.n_samples)
    log.info("Sampled %d records", len(samples))

    # -- Feature specs -------------------------------------------------------
    feature_specs = _build_feature_specs(schema, args.features)
    log.info("Feature specs: %d", len(feature_specs))

    # -- Run solves ----------------------------------------------------------
    project_root = str(Path(__file__).resolve().parent)
    work_items = [(rec, feature_specs, args.timeout, project_root) for rec in samples]

    sample_results: list[dict | None] = [None] * len(samples)
    t_start = time.time()

    if args.jobs == 1:
        for i, item in enumerate(work_items):
            sample_results[i] = _process_sample(item)
            done = i + 1
            elapsed = time.time() - t_start
            rate = done / elapsed
            eta = (len(samples) - done) / rate if rate > 0 else float("nan")
            if done % 10 == 0 or done == len(samples):
                n_ok = sum(1 for r in sample_results[:done] if r is not None)
                log.info("Sample %d/%d  ok=%d  eta=%.0fs", done, len(samples), n_ok, eta)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_idx = {executor.submit(_process_sample, item): i for i, item in enumerate(work_items)}
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                sample_results[idx] = future.result()
                done += 1
                elapsed = time.time() - t_start
                rate = done / elapsed
                eta = (len(samples) - done) / rate if rate > 0 else float("nan")
                if done % 10 == 0 or done == len(samples):
                    n_ok = sum(1 for r in sample_results if r is not None)
                    log.info("Sample %d/%d  ok=%d  eta=%.0fs", done, len(samples), n_ok, eta)

    # -- Aggregate -----------------------------------------------------------
    rows = _aggregate(sample_results, feature_specs)
    rows.sort(key=lambda r: r["mean_info_gain"] if not math.isnan(r["mean_info_gain"]) else -1, reverse=True)

    # -- Write ---------------------------------------------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "feature_key", "feature_label", "mean_info_gain", "std_info_gain",
        "n_valid", "n_timeout", "n_skipped_baseline_zero",
        "n_skipped_constrained_zero", "n_skipped_pred_none",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info("Written: %s", output_path)

    valid = [r for r in rows if not math.isnan(r["mean_info_gain"]) and r["n_valid"] > 0]
    if valid:
        log.info("Top 10 features by information gain:")
        for r in valid[:10]:
            log.info("  %.4f  %s  (n=%d)", r["mean_info_gain"], r["feature_label"], r["n_valid"])


if __name__ == "__main__":
    main()
