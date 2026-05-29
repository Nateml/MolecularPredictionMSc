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
import torch.nn.functional as F

# Project root on path so asp_bridge is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config
from data.datasets import dataset_from_schema, RawFeatureDataset
from model.model import MultitaskMLP, MultitaskMLPWithQRegression
from model.conformal import sets_from_outputs
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS_CONTINUOUS
from predicates.interface import PredictionInterface, PredictionEntry
from predicates.compiler import compile_asp_constraint
from asp.bridge import solve
from model.loader import load_model, load_thresholds

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps continuous feature names (as they appear in the schema) to the
# atom-count keys expected by PredictionInterface / ASP solver.
_ELEMENT_TO_COUNT_KEY: dict[str, str] = {
    "element_C": "num_carbons",
    "element_H": "num_hydrogens",
    "element_N": "num_nitrogens",
    "element_O": "num_oxygens",
    "element_P": "num_phosphorus",
    "element_S": "num_sulfurs",
}

_REQUIRED_COUNT_KEYS = {"num_carbons", "num_hydrogens", "num_nitrogens", "num_oxygens"}
_ALL_COUNT_KEYS = {"num_carbons", "num_hydrogens", "num_nitrogens",
                   "num_oxygens", "num_phosphorus", "num_sulfurs"}

# Reverse lookup: predicate name → FEATURE_VECTOR_DEFINITIONS integer key.
_PREDICATE_TO_FVD_KEY: dict[str, int] = {
    v["predicate"]: k for k, v in FEATURE_VECTOR_DEFINITIONS.items()
}

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

def _binary_predictions(
    logits: torch.Tensor,
    threshold: float,
) -> dict:
    """Convert binary head logits → MACCS key entries.

    Only includes keys that have an ASP definition and where the model
    confidence exceeds `threshold`.
    """
    probs = torch.sigmoid(logits).tolist()
    entries = {}
    for i, p in enumerate(probs):
        if i not in MACCS_DEFINITIONS:
            continue
        if p >= threshold:
            entries[i] = PredictionEntry(pred_set=[1], proba=[p])
        elif p <= 1 - threshold:
            entries[i] = PredictionEntry(pred_set=[0], proba=[1 - p])
        # uncertain → no entry → no constraint
    return entries


def _categorical_predictions(
    logits_list: list[torch.Tensor],
    feature_order: list[str],
    feature_edges: dict[str, list[float]],
    threshold: float,
) -> dict:
    """Convert categorical head logits → structural feature entries.

    Maps predicted bin indices back to FEATURE_VECTOR_DEFINITIONS keys
    (e.g., 'aldehyde' argmax=1 → predicate 'aldehyde_bin_1' → key 172).
    """
    entries = {}
    for logits, feature in zip(logits_list, feature_order):
        probs = F.softmax(logits, dim=-1).tolist()
        class_idx = int(torch.argmax(logits).item())
        confidence = probs[class_idx]
        if confidence < threshold:
            continue
        predicate_name = f"{feature}_bin_{class_idx}"
        key = _PREDICATE_TO_FVD_KEY.get(predicate_name)
        if key is None:
            continue
        entries[key] = PredictionEntry(pred_set=[1], proba=[probs[class_idx]])
    return entries


def _atom_counts_from_quantiles(
    quantile_list: list[torch.Tensor],
    feature_order: list[str],
    cont_mean: torch.Tensor | None,
    cont_std: torch.Tensor | None,
) -> dict[str, int]:
    """Extract atom counts from quantile head outputs.

    Takes the median quantile for element_* features, inverts z-score normalization
    if stats are available, then rounds to the nearest non-negative integer.
    """
    counts = {k: 0 for k in _ALL_COUNT_KEYS}
    mid = len(quantile_list[0].tolist()) // 2  # median quantile index
    for feat_idx, (quantiles, feature) in enumerate(zip(quantile_list, feature_order)):
        count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
        if count_key is None:
            continue
        val = quantiles[mid].item()
        if cont_std is not None and cont_mean is not None:
            val = val * cont_std[feat_idx].item() + cont_mean[feat_idx].item()
        counts[count_key] = max(0, round(val))
    return counts


def _atom_count_ranges_from_quantiles(
    quantile_list: list[torch.Tensor],
    feature_order: list[str],
    cont_mean: torch.Tensor | None,
    cont_std: torch.Tensor | None,
) -> dict[str, tuple[int, int]]:
    """Return inclusive integer [lo, hi] ranges for each atom count from quantile outputs.

    Uses the lowest and highest quantile outputs as the bounds, unnormalizing
    with z-score stats when available.
    """
    ranges: dict[str, tuple[int, int]] = {k: (0, 0) for k in _ALL_COUNT_KEYS}
    for feat_idx, (quantiles, feature) in enumerate(zip(quantile_list, feature_order)):
        count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
        if count_key is None:
            continue
        q_vals = quantiles.tolist()
        lo_val, hi_val = q_vals[0], q_vals[-1]
        if cont_std is not None and cont_mean is not None:
            m = cont_mean[feat_idx].item()
            s = cont_std[feat_idx].item()
            lo_val = lo_val * s + m
            hi_val = hi_val * s + m
        lo = max(0, round(lo_val))
        hi = max(lo, round(hi_val))
        ranges[count_key] = (lo, hi)
    return ranges


def _atom_counts_from_categorical(
    logits_list: list[torch.Tensor],
    feature_order: list[str],
) -> dict[str, int]:
    """Extract atom counts from categorical head argmax (for PackedDataset).

    Assumes class index == atom count (standard for integer-spaced edges).
    """
    counts = {k: 0 for k in _ALL_COUNT_KEYS}
    for logits, feature in zip(logits_list, feature_order):
        count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
        if count_key is None:
            continue
        counts[count_key] = int(torch.argmax(logits).item())
    return counts


def _structural_entries(outputs: dict, dataset, threshold: float) -> dict:
    """Binary + categorical structural entries, without atom counts."""
    entries: dict = {}
    entries.update(_binary_predictions(outputs["binary"].squeeze(0), threshold))
    if isinstance(dataset, RawFeatureDataset):
        cat_logits = [t.squeeze(0) for t in outputs["categorical"]]
        entries.update(_categorical_predictions(
            cat_logits,
            dataset.onehot_feature_order,
            dataset.onehot_feature_edges,
            threshold,
        ))
    return entries


def _continuous_range_constraints(
    lo_vals: dict[str, float],
    hi_vals: dict[str, float],
) -> list[str]:
    """ASP numeric range rules from denormalized continuous feature bounds."""
    rules = []
    for feature, lo_val in lo_vals.items():
        entry = FEATURE_VECTOR_DEFINITIONS_CONTINUOUS.get(feature)
        if entry is None:
            continue
        pred = entry["predicate"]
        lo = max(0, round(lo_val))
        hi = max(lo, round(hi_vals[feature]))
        rules.append(entry["definition"])
        rules.append(f":- {pred}(N), N < {lo}.")
        rules.append(f":- {pred}(N), N > {hi}.")
    return rules


def _continuous_constraints_from_median(
    quantile_list: list[torch.Tensor],
    feature_order: list[str],
    cont_mean: torch.Tensor | None,
    cont_std: torch.Tensor | None,
) -> list[str]:
    mid = len(quantile_list[0].tolist()) // 2
    lo_vals: dict[str, float] = {}
    for feat_idx, (quantiles, feature) in enumerate(zip(quantile_list, feature_order)):
        val = quantiles[mid].item()
        if cont_std is not None and cont_mean is not None:
            val = val * cont_std[feat_idx].item() + cont_mean[feat_idx].item()
        lo_vals[feature] = val
    return _continuous_range_constraints(lo_vals, lo_vals)


def _continuous_constraints_from_intervals(
    quantile_sets: torch.Tensor,
    feature_order: list[str],
    cont_mean: torch.Tensor | None,
    cont_std: torch.Tensor | None,
) -> list[str]:
    lo_vals: dict[str, float] = {}
    hi_vals: dict[str, float] = {}
    for feat_idx, feature in enumerate(feature_order):
        if feat_idx >= quantile_sets.shape[0]:
            break
        lo_n = quantile_sets[feat_idx, 0].item()
        hi_n = quantile_sets[feat_idx, 1].item()
        if cont_std is not None and cont_mean is not None:
            m = cont_mean[feat_idx].item()
            s = cont_std[feat_idx].item()
            lo_n = lo_n * s + m
            hi_n = hi_n * s + m
        lo_vals[feature] = lo_n
        hi_vals[feature] = hi_n
    return _continuous_range_constraints(lo_vals, hi_vals)




def _conformal_structural_entries(sets: dict, dataset) -> dict:
    """Binary + categorical structural entries derived from conformal prediction sets.

    Singleton sets → hard constraint. Ambiguous sets (both classes included) → no constraint.
    """
    entries: dict = {}

    binary_sets = sets["binary"].squeeze(0)  # (num_binary, 2)
    for i in range(binary_sets.shape[0]):
        if i not in MACCS_DEFINITIONS:
            continue
        inc_0 = binary_sets[i, 0].item()
        inc_1 = binary_sets[i, 1].item()
        if inc_1 and not inc_0:
            entries[i] = PredictionEntry(pred_set=[1], proba=[1.0])
        elif inc_0 and not inc_1:
            entries[i] = PredictionEntry(pred_set=[0], proba=[1.0])

    if isinstance(dataset, RawFeatureDataset):
        for cat_set, feature in zip(sets["categorical"], dataset.onehot_feature_order):
            included = cat_set.squeeze(0).nonzero(as_tuple=False).squeeze(-1).tolist()
            if len(included) == 1:
                predicate_name = f"{feature}_bin_{included[0]}"
                key = _PREDICATE_TO_FVD_KEY.get(predicate_name)
                if key is not None:
                    entries[key] = PredictionEntry(pred_set=[1], proba=[1.0])

    return entries


def _conformal_atom_count_ranges(
    sets: dict,
    feature_order: list[str],
    cont_mean: torch.Tensor | None,
    cont_std: torch.Tensor | None,
) -> dict[str, tuple[int, int]]:
    """Atom count [lo, hi] ranges from conformal quantile intervals.

    Intervals are in normalized space; unnormalizes before rounding.
    """
    ranges: dict[str, tuple[int, int]] = {k: (0, 0) for k in _ALL_COUNT_KEYS}
    if sets["quantile"] is None:
        return ranges
    quantile_sets = sets["quantile"].squeeze(0)  # (F, 2)
    for feat_idx, feature in enumerate(feature_order):
        count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
        if count_key is None:
            continue
        lo_norm = quantile_sets[feat_idx, 0].item()
        hi_norm = quantile_sets[feat_idx, 1].item()
        if cont_std is not None and cont_mean is not None:
            m = cont_mean[feat_idx].item()
            s = cont_std[feat_idx].item()
            lo_val = lo_norm * s + m
            hi_val = hi_norm * s + m
        else:
            lo_val, hi_val = lo_norm, hi_norm
        lo = max(0, round(lo_val))
        hi = max(lo, round(hi_val))
        ranges[count_key] = (lo, hi)
    return ranges


def predictions_to_interface(
    outputs: dict,
    dataset,
    threshold: float,
) -> PredictionInterface:
    """Convert raw model outputs to a PredictionInterface (single median-count combination)."""
    entries = _structural_entries(outputs, dataset, threshold)
    if isinstance(dataset, RawFeatureDataset):
        q_logits = [t.squeeze(0) for t in outputs["quantile"]]
        counts = _atom_counts_from_quantiles(
            q_logits, dataset.continuous_feature_order,
            cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
        )
    else:
        cat_logits = [t.squeeze(0) for t in outputs["categorical"]]
        counts = _atom_counts_from_categorical(cat_logits, dataset.onehot_feature_order)
    entries.update(counts)
    return PredictionInterface.model_validate(entries)


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
        # Fall back to latest checkpoint
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
        if args.spectrum_file == "-":
            raw_spectrum = sys.stdin.read()
        else:
            raw_spectrum = Path(args.spectrum_file).read_text(encoding="utf-8")
    elif args.spectrum:
        raw_spectrum = args.spectrum
    else:
        print("Error: provide --spectrum or --spectrum-file.", file=sys.stderr)
        sys.exit(1)

    spectrum_str = parse_spectrum(raw_spectrum)

    # Preprocess spectrum
    x = dataset._spectrum_to_dense(spectrum_str).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(x)

    is_raw = isinstance(dataset, RawFeatureDataset)

    if thresholds is not None:
        print("Using conformal prediction sets to derive constraints...")
        q_low_idx = 0
        q_high_idx = model.num_quantiles - 1
        conf_sets = sets_from_outputs(outputs, thresholds, q_low_idx, q_high_idx)
        structural = _conformal_structural_entries(conf_sets, dataset)
        if is_raw and conf_sets["quantile"] is not None:
            count_ranges = _conformal_atom_count_ranges(
                conf_sets, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
        elif is_raw:
            # Fallback if for some reasong there are no conformal calibrated quantile sets
            q_logits = [t.squeeze(0) for t in outputs["quantile"]]
            count_ranges = _atom_count_ranges_from_quantiles(
                q_logits, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
        else:
            count_ranges = None
    else:
        print("Warning: no conformal thresholds found; using manual thresholding of raw outputs. ")
        structural = _structural_entries(outputs, dataset, threshold=args.threshold)
        count_ranges = None
        if is_raw:
            q_logits = [t.squeeze(0) for t in outputs["quantile"]]
            count_ranges = _atom_count_ranges_from_quantiles(
                q_logits, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )

    # Continuous feature range constraints — computed once, appended to every combo's asp_constraints
    if is_raw:
        if thresholds is not None and conf_sets.get("quantile") is not None:
            extra_constraints = _continuous_constraints_from_intervals(
                conf_sets["quantile"].squeeze(0),
                dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
        else:
            q_all = [t.squeeze(0) for t in outputs["quantile"]]
            extra_constraints = _continuous_constraints_from_median(
                q_all, dataset.continuous_feature_order,
                cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
            )
    else:
        extra_constraints = []

    if args.use_median_counts and is_raw:
        q_logits = [t.squeeze(0) for t in outputs["quantile"]]
        single_counts = _atom_counts_from_quantiles(
            q_logits, dataset.continuous_feature_order,
            cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
        )
        print(f"Predicted formula (median): C{single_counts.get('num_carbons', 0)}"
              f"H{single_counts.get('num_hydrogens', 0)}"
              f"N{single_counts.get('num_nitrogens', 0)}"
              f"O{single_counts.get('num_oxygens', 0)}")
        combo_iter = iter([single_counts])
        n_combos = 1
    elif is_raw and count_ranges is not None:
        count_keys = list(count_ranges.keys())
        n_combos = 1
        for lo, hi in count_ranges.values():
            n_combos *= (hi - lo + 1)
        combo_iter = (
            dict(zip(count_keys, combo))
            for combo in itertools.product(
                *[range(lo, hi + 1) for lo, hi in count_ranges.values()]
            )
        )
        print("Atom count ranges:")
        for k, (lo, hi) in count_ranges.items():
            print(f"  {k}: [{lo}, {hi}]")
    else:
        cat_logits = [t.squeeze(0) for t in outputs["categorical"]]
        single_counts = _atom_counts_from_categorical(cat_logits, dataset.onehot_feature_order)
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
        interface = PredictionInterface.model_validate({**structural, **count_dict})
        asp_constraints, atom_counts = compile_asp_constraint(interface)
        asp_constraints = asp_constraints + extra_constraints
        if args.show_constraints:
            print(f"\n  formula: C{atom_counts.get('num_carbons',0)}"
                  f"H{atom_counts.get('num_hydrogens',0)}"
                  f"N{atom_counts.get('num_nitrogens',0)}"
                  f"O{atom_counts.get('num_oxygens',0)}")
            for c in asp_constraints:
                print(f"    {c}")
        cands, _ = solve(asp_constraints, atom_counts, output_list_of_smiles=True, timeout=args.solve_timeout, debug=args.asp_debug)
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
