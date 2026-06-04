#!/usr/bin/env python
"""
Spectrum → candidate SMILES.

Runs the trained model on an EI-MS spectrum, converts predictions into
ASP constraints, and enumerates candidate molecules.

Usage:
    python infer.py --spectrum "100:0.5 200:0.3 300:1.0"
    python infer.py --spectrum "..." --checkpoint models/multitask_mlp.best.pt
    python infer.py --spectrum "..." --threshold 0.7 --max-candidates 20

Spectrum format: space-separated "mz:intensity" pairs (intensities need not
be normalised — the model normalises internally).
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import torch

# Project root on path so asp_bridge is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import config
from data.datasets import dataset_from_schema, RawFeatureDataset
from model.model import MultitaskMLP, MultitaskMLPWithQRegression
from model.conformal import sets_from_outputs
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS_CONTINUOUS
from predicates.interface import PredictionInterface, PredictionEntry
from predicates.compiler import compile_asp_constraint
from asp.bridge import solve
from model.loader import load_model, load_thresholds
from model.predict import predict_single

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMBINATION_WARNING_THRESHOLD = 100


# ---------------------------------------------------------------------------
# Spectrum parsing
# ---------------------------------------------------------------------------

def parse_spectrum(text: str) -> str:
    """Normalise spectrum text to "mz:intensity ..." format.

    Accepts:
      - Native format:   "100:0.5 200:1.0 ..."  (pass-through)
      - MassBank/NIST:   whitespace-separated rows "mz intensity [rel_intensity]"
                         with optional "PK$PEAK: ..." header line
    Returns space-joined "mz:intensity" pairs.
    """
    text = text.strip()
    if ":" in text.split()[0]:
        return text  # already native format

    pairs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.upper().startswith("PK$"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            mz = float(parts[0])
            intensity = float(parts[1])
        except ValueError:
            continue
        pairs.append(f"{mz}:{intensity}")
    if not pairs:
        raise ValueError("No valid peaks found in spectrum input.")
    return " ".join(pairs)


# ---------------------------------------------------------------------------
# Prediction → PredictionInterface conversion
# ---------------------------------------------------------------------------


def predictions_to_interface(
    outputs: dict,
    dataset,
    threshold: float,
) -> PredictionInterface:
    """Convert raw model outputs to a PredictionInterface (single median-count combination).

    Uses the median quantile output for element counts (RawFeatureDataset) or
    categorical argmax (PackedDataset). Structural entries are threshold-filtered.
    """
    from model.predict import (
        _ELEMENT_TO_COUNT_KEY,
        _structural_from_threshold,
        _count_ranges_from_categorical,
    )

    is_raw = isinstance(dataset, RawFeatureDataset)
    onehot_order = dataset.onehot_feature_order if is_raw else []
    cat_logits = [t.squeeze(0) for t in outputs.get("categorical", [])]

    structural = _structural_from_threshold(
        outputs["binary"].squeeze(0), cat_logits, onehot_order, threshold
    )

    if is_raw:
        q_list = [t.squeeze(0) for t in outputs.get("quantile", [])]
        mid = len(q_list[0].tolist()) // 2 if q_list else 0
        counts: dict[str, int] = {}
        for feat_idx, (q_tensor, feature) in enumerate(
            zip(q_list, dataset.continuous_feature_order)
        ):
            count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
            if count_key is None:
                continue
            val = q_tensor[mid].item()
            if dataset._cont_std is not None and dataset._cont_mean is not None:
                val = val * dataset._cont_std[feat_idx].item() + dataset._cont_mean[feat_idx].item()
            counts[count_key] = max(0, round(val))
    else:
        count_ranges = _count_ranges_from_categorical(cat_logits, onehot_order)
        counts = {k: lo for k, (lo, _) in count_ranges.items()}

    return PredictionInterface.model_validate({**structural, **counts})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Spectrum → candidate SMILES via neural net + ASP solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--spectrum", default=None,
        help='Space-separated "mz:intensity" pairs, e.g. "100:0.5 200:1.0". '
             'Also accepts MassBank/NIST whitespace-separated "mz intensity" rows.',
    )
    p.add_argument(
        "--spectrum-file", default=None, metavar="FILE",
        help='Read spectrum from FILE (use "-" for stdin). '
             'Accepts MassBank copy-paste (mz intensity rows) or native "mz:intensity" format.',
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        help="Path to model checkpoint (.pt). Defaults to models/multitask_mlp.best.pt",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Confidence threshold for including a binary/categorical prediction as a constraint (default: 0.5)",
    )
    p.add_argument(
        "--use-median-counts", action="store_true",
        help="Use median quantile predictions for atom counts instead of categorical head argmax (only applies to RawFeatureDataset models)",
    )
    p.add_argument(
        "--max-candidates", type=int, default=None,
        help="Maximum number of candidate SMILES to return (default: all)",
    )
    p.add_argument(
        "--show-constraints", action="store_true",
        help="Print compiled ASP constraints before solving",
    )
    p.add_argument(
        "--solve-timeout", type=int, default=config.solver.timeout,
        help=f"Wall-clock time limit in seconds for each ASP solve call (0 = no limit, default: {config.solver.timeout})",
    )
    p.add_argument(
        "--asp-debug", action="store_true",
        help="passed debug=True to asp.bridge.solve()"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        config.checkpointing.resolved_save_dir() / "multitask_mlp.best.pt"
    )
    if not checkpoint_path.exists():
        checkpoint_path = config.checkpointing.resolved_save_dir() / "multitask_mlp.latest.pt"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}. Train the model first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {checkpoint_path}...")
    model, dataset, device = load_model(checkpoint_path)

    thresholds = load_thresholds(checkpoint_path)
    if thresholds is not None:
        print(f"Conformal thresholds loaded (alpha={config.conformal.alpha}).")
    else:
        print(f"No conformal thresholds found; using manual threshold={args.threshold}.")

    # Load spectrum
    if args.spectrum_file:
        raw_spectrum = sys.stdin.read() if args.spectrum_file == "-" else Path(args.spectrum_file).read_text(encoding="utf-8")
    elif args.spectrum:
        raw_spectrum = args.spectrum
    else:
        print("Error: provide --spectrum or --spectrum-file.", file=sys.stderr)
        sys.exit(1)

    x = dataset._spectrum_to_dense(parse_spectrum(raw_spectrum))

    pred = predict_single(x, model, dataset, threshold=args.threshold, thresholds=thresholds)
    is_raw = isinstance(dataset, RawFeatureDataset)

    # Continuous range entries to include in every PredictionInterface combo
    cont_entries = {f: ContinuousRange(lo=lo, hi=hi) for f, (lo, hi) in pred.continuous_ranges.items()}

    # Build formula combinations from predicted element count ranges
    count_ranges = pred.count_ranges
    if args.use_median_counts and is_raw:
        single_counts = {k: (lo + hi) // 2 for k, (lo, hi) in count_ranges.items()}
        print(f"Predicted formula (median): C{single_counts.get('num_carbons', 0)}"
              f"H{single_counts.get('num_hydrogens', 0)}"
              f"N{single_counts.get('num_nitrogens', 0)}"
              f"O{single_counts.get('num_oxygens', 0)}"
              f"S{single_counts.get('num_sulfurs', 0)}"
              f"P{single_counts.get('num_phosphorus', 0)}")
        combo_iter = iter([single_counts])
        n_combos = 1
    elif is_raw:
        n_combos = 1
        for lo, hi in count_ranges.values():
            n_combos *= (hi - lo + 1)
        combo_iter = (
            dict(zip(count_ranges, combo))
            for combo in itertools.product(*[range(lo, hi + 1) for lo, hi in count_ranges.values()])
        )
        print("Atom count ranges:")
        for k, (lo, hi) in count_ranges.items():
            print(f"  {k}: [{lo}, {hi}]")
    else:
        single_counts = {k: lo for k, (lo, _) in count_ranges.items()}
        if all(v == 0 for v in single_counts.values()):
            print("Warning: all atom counts predicted as 0. Model may not be trained yet.", file=sys.stderr)
        print(f"Predicted formula: C{single_counts.get('num_carbons', 0)}"
              f"H{single_counts.get('num_hydrogens', 0)}"
              f"N{single_counts.get('num_nitrogens', 0)}"
              f"O{single_counts.get('num_oxygens', 0)}")
        combo_iter = iter([single_counts])
        n_combos = 1

    print(f"{n_combos} formula combination(s) to solve." if n_combos > 1 else "Using single predicted formula.")
    if n_combos > COMBINATION_WARNING_THRESHOLD:
        print(f"Warning: {n_combos} formula combinations to solve. This may take a long time.", file=sys.stderr)
        try:
            with open("/dev/tty") as tty:
                print("Do you want to continue? [y/N] ", end="", file=sys.stderr, flush=True)
                choice = tty.readline().strip().lower()
        except OSError:
            print("Non-interactive mode; continuing automatically.", file=sys.stderr)
            choice = "y"
        if choice != "y":
            print("Aborting.")
            sys.exit(0)

    all_candidates: list[str] = []
    seen: set[str] = set()
    for count_dict in combo_iter:
        interface = PredictionInterface.model_validate({**pred.structural, **count_dict, **cont_entries})
        asp_constraints, atom_counts = compile_asp_constraint(interface)
        if args.show_constraints:
            print(f"\n  formula: C{atom_counts.get('num_carbons',0)}"
                  f"H{atom_counts.get('num_hydrogens',0)}"
                  f"N{atom_counts.get('num_nitrogens',0)}"
                  f"O{atom_counts.get('num_oxygens',0)}")
            for c in asp_constraints:
                print(f"    {c}")
        cands, _, _ = solve(asp_constraints, atom_counts, output_list_of_smiles=True, timeout=args.solve_timeout, debug=args.asp_debug)
        for smi in cands:
            if smi not in seen:
                seen.add(smi)
                all_candidates.append(smi)

    if args.max_candidates:
        all_candidates = all_candidates[:args.max_candidates]

    print(f"\nFound {len(all_candidates)} unique candidate(s) across {n_combos} formula(s):")
    for i, smi in enumerate(all_candidates, 1):
        print(f"  {i}. {smi}")


if __name__ == "__main__":
    main()
