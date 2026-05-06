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
    """
    data_path = config.data.resolved_dataset_path()
    schema_path = config.data.resolved_schema_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = dataset_from_schema(data_path, schema_path,
                                  spectrum_len=config.model.spectrum_length)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Build model from checkpoint metadata so architecture matches exactly
    # regardless of current config.yaml values.
    # shape matches exactly regardless of current config.yaml values.
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
) -> dict[str, int]:
    """Extract atom counts from quantile head outputs.

    Takes the median quantile (middle index) for element_* features and
    rounds to the nearest non-negative integer.
    """
    counts = {k: 0 for k in _ALL_COUNT_KEYS}
    mid = len(quantile_list[0].tolist()) // 2  # median quantile index
    for quantiles, feature in zip(quantile_list, feature_order):
        count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
        if count_key is None:
            continue
        counts[count_key] = max(0, round(quantiles[mid].item()))
    return counts


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


def predictions_to_interface(
    outputs: dict,
    dataset,
    threshold: float,
) -> PredictionInterface:
    """Convert raw model outputs to a PredictionInterface."""
    entries: dict = {}

    # MACCS binary features
    entries.update(_binary_predictions(outputs["binary"].squeeze(0), threshold))

    is_raw = isinstance(dataset, RawFeatureDataset)

    # Structural categorical features
    cat_logits = [t.squeeze(0) for t in outputs["categorical"]]
    if is_raw:
        entries.update(_categorical_predictions(
            cat_logits,
            dataset.onehot_feature_order,
            dataset.onehot_feature_edges,
            threshold,
        ))
        # Atom counts from quantile heads
        q_logits = [t.squeeze(0) for t in outputs["quantile"]]
        counts = _atom_counts_from_quantiles(q_logits, dataset.continuous_feature_order)
    else:
        counts = _atom_counts_from_categorical(cat_logits, dataset.onehot_feature_order)

    # Require all four base atom counts; warn and default to 0 if missing
    for k in _REQUIRED_COUNT_KEYS:
        if counts.get(k, 0) == 0 and k not in _ELEMENT_TO_COUNT_KEY.values():
            pass  # already 0
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

    # Convert to PredictionInterface
    interface = predictions_to_interface(outputs, dataset, threshold=args.threshold)

    # Compile ASP constraints
    asp_constraints, atom_counts = compile_asp_constraint(interface)

    if args.show_constraints:
        print("\nAtom counts:", atom_counts)
        print("\nASP constraints:")
        for c in asp_constraints:
            print(" ", c)
        print()

    if all(v == 0 for v in atom_counts.values()):
        print("Warning: all atom counts predicted as 0. Model may not be trained yet.", file=sys.stderr)

    print(f"Predicted formula: C{atom_counts.get('num_carbons', 0)}"
          f"H{atom_counts.get('num_hydrogens', 0)}"
          f"N{atom_counts.get('num_nitrogens', 0)}"
          f"O{atom_counts.get('num_oxygens', 0)}")
    print("Solving...")

    candidates, n = solve(asp_constraints, atom_counts, output_list_of_smiles=True)

    if args.max_candidates:
        candidates = candidates[:args.max_candidates]

    print(f"\nFound {n} candidate(s):")
    for i, smi in enumerate(candidates, 1):
        print(f"  {i}. {smi}")


if __name__ == "__main__":
    main()
