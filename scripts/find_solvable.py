#!/usr/bin/env python3
"""
Pre-screen a dataset split to find molecules whose ground-truth formula is
solvable by the ASP solver within a given timeout.

Runs an *unconstrained* ASP solve on the true molecular formula for each
record.  If even this baseline times out, the formula is too hard for the
full pipeline.  The solvable indices are saved as a split .pkl so they can
be passed directly to evaluate_molecules.py via --split-pkl.

Progress is checkpointed every --checkpoint-every records so the run can be
safely interrupted and resumed: on restart the script skips already-screened
indices and continues from where it left off.

Usage
-----
    # Screen a split pkl:
    python find_solvable.py \\
        --split-pkl data/splits/<hash>/test.pkl \\
        --full-dataset data/splits/<hash>/dataset.pkl \\
        --timeout 5

    # Screen a JSON test set:
    python find_solvable.py --test-set data/multitask_mlp_hardcoded.test.json --timeout 5

    # Resume an interrupted run (auto-detected from progress file):
    python find_solvable.py --split-pkl data/splits/<hash>/test.pkl \\
        --full-dataset data/splits/<hash>/dataset.pkl --timeout 5

Output
------
<output>.pkl         — list of solvable dataset indices (same format as split pkls)
<output>.json        — summary: n_screened, n_solvable, timeout, reject reasons
<output>.progress    — incremental checkpoint (deleted on clean finish)
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import signal
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from asp.bridge import solve, is_asp_supported
from filters import formula_complexity_filter
from predicates.interface import PredictionInterface, PredictionEntry, ContinuousRange
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS_CONTINUOUS
from predicates.compiler import compile_asp_constraint

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── atom count / GT label extraction ─────────────────────────────────────────

def _gt_atom_counts(record: dict) -> dict | None:
    sf = record.get("structural_features")
    if not sf:
        return None
    try:
        return {
            "num_carbons":    int(sf["element_C"]),
            "num_hydrogens":  int(sf["element_H"]),
            "num_oxygens":    int(sf["element_O"]),
            "num_nitrogens":  int(sf["element_N"]),
            "num_sulfurs":    int(sf["element_S"]),
            "num_phosphorus": int(sf["element_P"]),
        }
    except KeyError:
        return None


def _gt_full_interface(record: dict) -> PredictionInterface | None:
    """Build a PredictionInterface from ground-truth labels.

    Includes exact MACCS binary constraints and exact continuous structural
    feature constraints in addition to atom counts.  Returns None when required
    data is missing.
    """
    counts = _gt_atom_counts(record)
    if counts is None:
        return None

    entries: dict = {**counts}

    maccs = record.get("maccs")
    if maccs is not None:
        for key_id in MACCS_DEFINITIONS:
            if key_id < len(maccs):
                entries[key_id] = PredictionEntry(pred_set=[int(maccs[key_id])])

    sf = record.get("structural_features") or {}
    for feat_name in FEATURE_VECTOR_DEFINITIONS_CONTINUOUS:
        if feat_name in sf:
            val = float(sf[feat_name])
            entries[feat_name] = ContinuousRange(lo=val, hi=val)

    try:
        return PredictionInterface.model_validate(entries)
    except Exception as exc:
        log.debug(f"Failed to build GT interface: {exc}")
        return None


# ── checkpointing ─────────────────────────────────────────────────────────────

def _load_progress(progress_path: Path) -> dict[str, str]:
    """Load progress dict {str(dataset_idx): 'solvable'|'rejected'|'timeout'|'error'|'missing'}."""
    if not progress_path.exists():
        return {}
    try:
        with open(progress_path) as f:
            return json.load(f)
    except Exception as exc:
        log.warning(f"Could not read progress file {progress_path}: {exc} — starting fresh.")
        return {}


def _save_progress(progress_path: Path, progress: dict[str, str]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = progress_path.with_suffix(".progress.tmp")
    with open(tmp, "w") as f:
        json.dump(progress, f)
    tmp.replace(progress_path)


# ── dataset loading ───────────────────────────────────────────────────────────

def _load_records_with_indices(args: argparse.Namespace) -> list[tuple[int, dict]]:
    """Return list of (dataset_index, record_dict) pairs to screen."""
    if args.test_set:
        from config import config
        from data.datasets import dataset_from_schema

        schema_path = (
            Path(args.schema) if args.schema
            else config.data.resolved_schema_path()
        )
        ds = dataset_from_schema(
            Path(args.test_set), schema_path,
            spectrum_len=config.model.spectrum_length,
        )
        pairs = []
        for i in range(len(ds)):
            if args.limit and len(pairs) >= args.limit:
                break
            pairs.append((i, ds._load_record(i)))
        return pairs

    split_pkl = Path(args.split_pkl)
    if not split_pkl.exists():
        raise FileNotFoundError(f"Split pkl not found: {split_pkl}")

    full_path = Path(args.full_dataset)
    if not full_path.exists():
        raise FileNotFoundError(f"Full dataset not found: {full_path}")

    from config import config
    from data.datasets import dataset_from_schema

    schema_path = (
        Path(args.schema) if args.schema
        else config.data.resolved_schema_path()
    )
    ds = dataset_from_schema(
        full_path, schema_path,
        spectrum_len=config.model.spectrum_length,
    )

    with open(split_pkl, "rb") as f:
        indices: list[int] = pickle.load(f)

    log.info(f"Split has {len(indices)} indices.")

    pairs = []
    for idx in indices:
        if args.limit and len(pairs) >= args.limit:
            break
        pairs.append((idx, ds._load_record(idx)))
    return pairs


# ── screening ─────────────────────────────────────────────────────────────────

def screen(
    pairs: list[tuple[int, dict]],
    timeout: int,
    max_score: float | None,
    progress: dict[str, str],
    progress_path: Path,
    checkpoint_every: int,
    use_gt_labels: bool = False,
) -> tuple[list[int], dict]:
    """Screen records, skipping already-processed indices from progress.

    Returns (solvable_indices, stats_dict).
    """
    n_resumed = sum(1 for v in progress.values() if v != "pending")
    if n_resumed:
        log.info(f"Resuming: {n_resumed} records already processed.")

    # Counts over the *entire* run (resumed + new).
    def _tally(status: str) -> int:
        return sum(1 for v in progress.values() if v == status)

    interrupted = False

    def _handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        log.warning("Interrupted — saving progress and exiting cleanly.")

    signal.signal(signal.SIGINT, _handle_sigint)

    pending = [(idx, rec) for idx, rec in pairs if str(idx) not in progress]
    log.info(f"{len(pending)} records left to screen.")

    for i, (dataset_idx, record) in enumerate(tqdm(pending, desc="Screening")):
        if interrupted:
            break

        smiles = record.get("smiles", "")

        if max_score is not None and smiles:
            try:
                if not formula_complexity_filter(smiles, max_score=max_score):
                    progress[str(dataset_idx)] = "rejected"
                    continue
            except Exception:
                pass

        if smiles:
            supported, reason = is_asp_supported(smiles)
            if not supported:
                log.debug(f"[{dataset_idx}] ASP unsupported: {reason}")
                progress[str(dataset_idx)] = "unsupported"
                continue

        if use_gt_labels:
            interface = _gt_full_interface(record)
            if interface is None:
                progress[str(dataset_idx)] = "missing"
                continue
            asp_constraints, counts = compile_asp_constraint(interface)
        else:
            counts = _gt_atom_counts(record)
            if counts is None:
                progress[str(dataset_idx)] = "missing"
                continue
            asp_constraints = []

        try:
            _, n_cands, timed_out = solve(
                asp_constraints, counts,
                output_list_of_smiles=False,
                timeout=timeout,
                suppress_timeout_warning=True,
            )
            if timed_out:
                progress[str(dataset_idx)] = "timeout"
            elif n_cands == 0:
                progress[str(dataset_idx)] = "unsolvable"
            else:
                progress[str(dataset_idx)] = "solvable"
        except Exception as exc:
            log.debug(f"[{dataset_idx}] ASP error: {exc}")
            progress[str(dataset_idx)] = "error"

        if (i + 1) % checkpoint_every == 0:
            _save_progress(progress_path, progress)

    _save_progress(progress_path, progress)

    solvable = [int(k) for k, v in progress.items() if v == "solvable"]

    n_screened = len(progress)
    stats = {
        "n_screened":       n_screened,
        "n_solvable":       _tally("solvable"),
        "n_unsolvable":     _tally("unsolvable"),
        "n_unsupported":    _tally("unsupported"),
        "n_score_rejected": _tally("rejected"),
        "n_missing_counts": _tally("missing"),
        "n_timed_out":      _tally("timeout"),
        "n_error":          _tally("error"),
        "n_pending":        len(pairs) - n_screened,
        "timeout_s":        timeout,
        "max_score":        max_score,
        "use_gt_labels":    use_gt_labels,
        "complete":         not interrupted and len(pairs) == n_screened,
    }
    return solvable, stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Find ASP-solvable molecules in a dataset split.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--split-pkl", default=None, help="Path to a split indices .pkl file.")
    src.add_argument("--test-set", default=None, help="Path to a JSON dataset file.")

    p.add_argument(
        "--full-dataset", default=None,
        help="Path to full dataset .json file. Required with --split-pkl.",
    )
    p.add_argument("--schema", default=None, help="Path to schema JSON. Defaults to config value.")
    p.add_argument(
        "--timeout", type=int, default=5,
        help="Per-formula ASP wall-clock timeout in seconds (default: 5).",
    )
    p.add_argument(
        "--max-score", type=float, default=None,
        help="Optional formula_complexity_score pre-filter (default: disabled).",
    )
    p.add_argument("--limit", type=int, default=None, help="Screen only the first N records.")
    p.add_argument(
        "--output", default=None,
        help="Output .pkl path. Defaults to <input>_solvable_<timeout>s.pkl.",
    )
    p.add_argument(
        "--checkpoint-every", type=int, default=100,
        help="Save progress to disk every N records (default: 100).",
    )
    p.add_argument(
        "--use-gt-labels", action="store_true",
        help=(
            "Use the full ground-truth label set (MACCS keys + structural features) "
            "as ASP constraints in addition to atom counts. "
            "Without this flag only atom counts are constrained."
        ),
    )
    return p.parse_args()


def _default_output(args: argparse.Namespace) -> Path:
    label_tag = "_gt_labels" if args.use_gt_labels else ""
    stem = f"solvable{label_tag}_{args.timeout}s"
    if args.split_pkl:
        sp = Path(args.split_pkl)
        return sp.with_name(sp.stem + f"_{stem}.pkl")
    return Path(f"{stem}.pkl")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    if args.split_pkl and not args.full_dataset:
        print("ERROR: --full-dataset is required with --split-pkl", file=sys.stderr)
        sys.exit(1)

    out_pkl = Path(args.output) if args.output else _default_output(args)
    out_json = out_pkl.with_suffix(".json")
    progress_path = out_pkl.with_suffix(".progress")

    progress = _load_progress(progress_path)
    if progress:
        log.info(f"Loaded progress from {progress_path} ({len(progress)} entries).")

    log.info("Loading records ...")
    pairs = _load_records_with_indices(args)
    log.info(f"Loaded {len(pairs)} records.")

    solvable_indices, stats = screen(
        pairs, args.timeout, args.max_score,
        progress, progress_path, args.checkpoint_every,
        use_gt_labels=args.use_gt_labels,
    )

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(solvable_indices, f)

    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)

    if stats["complete"]:
        progress_path.unlink(missing_ok=True)
        log.info("Progress file deleted (run complete).")

    pct = 100.0 * stats["n_solvable"] / stats["n_screened"] if stats["n_screened"] else 0.0
    print(f"\n=== Solvability Screening ===")
    print(f"  Screened       : {stats['n_screened']}")
    print(f"  Solvable       : {stats['n_solvable']}  ({pct:.1f}%)")
    print(f"  Unsolvable     : {stats['n_unsolvable']}")
    print(f"  Unsupported    : {stats['n_unsupported']}")
    print(f"  Timed out      : {stats['n_timed_out']}")
    print(f"  Score-rejected : {stats['n_score_rejected']}")
    print(f"  Missing counts : {stats['n_missing_counts']}")
    print(f"  ASP errors     : {stats['n_error']}")
    if not stats["complete"]:
        print(f"  Still pending  : {stats['n_pending']}  (re-run to continue)")
    print(f"\n  Solvable pkl   : {out_pkl}")
    print(f"  Summary json   : {out_json}")
    if not stats["complete"]:
        print(f"  Progress file  : {progress_path}")
    print(f"\nUse with evaluate_molecules.py:")
    if args.split_pkl:
        print(f"  python evaluate_molecules.py \\")
        print(f"      --split-pkl {out_pkl} \\")
        print(f"      --full-dataset {args.full_dataset}")
    else:
        print(f"  python evaluate_molecules.py --split-pkl {out_pkl} --full-dataset <dataset>")


if __name__ == "__main__":
    main()
