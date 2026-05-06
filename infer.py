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
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS
from predicates.interface import PredictionInterface, PredictionEntry
from predicates.compiler import compile_asp_constraint
from asp_bridge import solve

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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: Path):
    """Return (model, dataset, device).

    The dataset is constructed purely for its schema attributes (maccs_length,
    feature orders, edges) — no records are loaded from disk.
    Applies saved continuous normalization stats to the dataset when present.
    """
    data_path = config.data.resolved_dataset_path()
    schema_path = config.data.resolved_schema_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = dataset_from_schema(data_path, schema_path,
                                  spectrum_len=config.model.spectrum_length)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Build model from checkpoint metadata so architecture matches exactly
    # regardless of current config.yaml values.
    if isinstance(dataset, RawFeatureDataset):
        model = MultitaskMLPWithQRegression(
            spectrum_len=config.model.spectrum_length,
            hidden_dims=config.model.hidden_dims,
            num_binary=ckpt.get("maccs_length", dataset.maccs_length),
            categorical_sizes=ckpt.get("categorical_sizes", dataset.categorical_sizes()),
            num_quantiles=ckpt.get("num_quantiles", config.model.num_quantiles),
            num_regression=ckpt.get("num_regression", len(dataset.continuous_feature_order)),
            regression_hidden_dim=config.model.regression_hidden_dim,
            dropout=config.model.dropout,
        )
        # Restore normalization so quantile outputs can be inverted
        cont_mean = ckpt.get("continuous_mean")
        cont_std = ckpt.get("continuous_std")
        if cont_mean is not None and cont_std is not None:
            dataset.normalise_continuous(cont_mean, cont_std)
    else:
        model = MultitaskMLP(
            spectrum_len=config.model.spectrum_length,
            hidden_dims=config.model.hidden_dims,
            num_binary=ckpt.get("maccs_length", dataset.maccs_length),
            categorical_sizes=ckpt.get("categorical_sizes", dataset.categorical_sizes()),
            dropout=config.model.dropout,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, dataset, device


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
        "--spectrum", required=True,
        help='Space-separated "mz:intensity" pairs, e.g. "100:0.5 200:1.0"',
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
        "--max-candidates", type=int, default=None,
        help="Maximum number of candidate SMILES to return (default: all)",
    )
    p.add_argument(
        "--show-constraints", action="store_true",
        help="Print compiled ASP constraints before solving",
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

    # Preprocess spectrum
    x = dataset._spectrum_to_dense(args.spectrum).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(x)

    # Build structural constraints (binary + categorical) shared across all count combos
    structural = _structural_entries(outputs, dataset, threshold=args.threshold)

    is_raw = isinstance(dataset, RawFeatureDataset)
    if is_raw:
        q_logits = [t.squeeze(0) for t in outputs["quantile"]]
        count_ranges = _atom_count_ranges_from_quantiles(
            q_logits, dataset.continuous_feature_order,
            cont_mean=dataset._cont_mean, cont_std=dataset._cont_std,
        )
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

    print(f"Solving {n_combos} formula combination(s)...")

    all_candidates: list[str] = []
    seen: set[str] = set()
    for count_dict in combo_iter:
        interface = PredictionInterface.model_validate({**structural, **count_dict})
        asp_constraints, atom_counts = compile_asp_constraint(interface)
        if args.show_constraints:
            print(f"\n  formula: C{atom_counts.get('num_carbons',0)}"
                  f"H{atom_counts.get('num_hydrogens',0)}"
                  f"N{atom_counts.get('num_nitrogens',0)}"
                  f"O{atom_counts.get('num_oxygens',0)}")
            for c in asp_constraints:
                print(f"    {c}")
        cands, _ = solve(asp_constraints, atom_counts, output_list_of_smiles=True)
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
