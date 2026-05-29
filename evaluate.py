#!/usr/bin/env python
"""
Evaluate a pretrained checkpoint on a test set.

Computes per-target accuracy, precision, recall, and F1 for all binary
(MACCS key) and categorical (structural feature) heads. Results are printed
to stdout and saved as CSV.

With --conformal, loads conformal_thresholds.pt from the checkpoint directory
and reports coverage rate, average set size, and singleton rate instead.

Usage:
    # Load from JSON file
    python evaluate.py --test-set data/test.json
    python evaluate.py --test-set data/test.json --checkpoint models/final/multitask_mlp_mixed_best.pt
    python evaluate.py --test-set data/test.json --threshold 0.5 --output results.csv
    python evaluate.py --test-set data/test.json --conformal
    python evaluate.py --test-set data/test.json --conformal --thresholds-path path/to/conformal_thresholds.pt
    
    # Load from saved splits
    python evaluate.py --from-splits --split-name test
    python evaluate.py --from-splits --split-name val --checkpoint models/final/multitask_mlp.best.pt
    python evaluate.py --from-splits --split-name train --conformal --output train_conformal_results.csv
    python evaluate.py --from-splits --split-name test --split-hash a3f2c1b9d4e5  # Specific split
"""
from __future__ import annotations

import argparse
import csv
import sys
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))  # Add this line

from config import config
from data.datasets import RawFeatureDataset
from data.split import splits_dir as get_splits_dir
from infer import load_model
from model.loader import load_model
from model.conformal import sets_from_outputs
import pickle


def load_dataset_from_splits(
    split_name: str = "test",
    data_path: Path | None = None,
    schema_path: Path | None = None,
    split_hash: str | None = None,
) -> tuple[RawFeatureDataset, dict]:
    """
    Load a dataset from saved splits.
    
    Args:
        split_name: "train", "val", "calibration", or "test"
        data_path: Path to data JSON. If None, uses config.data.resolved_dataset_path()
        schema_path: Path to schema. If None, uses config.data.resolved_schema_path()
        split_hash: Hash of split directory. If None, constructs from paths.
    
    Returns:
        (dataset_subset, split_info_dict)
    """
    if data_path is None:
        data_path = config.data.resolved_dataset_path()
    if schema_path is None:
        schema_path = config.data.resolved_schema_path()
    
    data_path = Path(data_path)
    schema_path = Path(schema_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema path not found: {schema_path}")
    
    # Construct splits directory
    fractions = {
        "train": 0.75,
        "val": 0.10,
        "calibration": 0.05,
        "test": 0.10,
    }
    split_hash_actual = split_hash or get_splits_dir(data_path, schema_path, fractions, seed=42)
    
    if isinstance(split_hash_actual, Path):
        split_dir = split_hash_actual
    else:
        split_dir = Path(__file__).resolve().parents[0] / "data" / "splits" / split_hash_actual
    
    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            f"Available splits can be found in {split_dir.parent}/"
        )
    
    # Load dataset object
    dataset_pkl = split_dir / "dataset.pkl"
    if not dataset_pkl.exists():
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_pkl}")
    
    with open(dataset_pkl, "rb") as f:
        dataset = pickle.load(f)
    
    # Load split indices
    split_pkl = split_dir / f"{split_name}.pkl"
    if not split_pkl.exists():
        raise FileNotFoundError(f"Split pickle not found: {split_pkl}")
    
    with open(split_pkl, "rb") as f:
        indices = pickle.load(f)
    
    # Create subset using indices
    from torch.utils.data import Subset
    subset = Subset(dataset, indices)
    
    # Load metadata
    meta_path = split_dir / "meta.json"
    split_info = {}
    if meta_path.exists():
        import json
        with open(meta_path, "r") as f:
            split_info = json.load(f)
    
    return subset, split_info


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--test-set", default=None,
        help="Path to test JSON dataset. Mutually exclusive with --from-splits.",
    )
    p.add_argument(
        "--from-splits", action="store_true",
        help="Load dataset from saved splits instead of JSON file.",
    )
    p.add_argument(
        "--split-name", default="test",
        help="Which split to load: 'train', 'val', 'calibration', or 'test' (default: test). "
             "Only used with --from-splits.",
    )
    p.add_argument(
        "--split-hash", default=None,
        help="Hash of the split directory. If None, constructs from config paths. "
             "Only used with --from-splits.",
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to .pt checkpoint. Defaults to models/multitask_mlp.best.pt",
    )
    p.add_argument(
        "--threshold", type=float, default=0.5,
        help="Sigmoid threshold for binary (MACCS) predictions (default: 0.5)",
    )
    p.add_argument(
        "--output", default="eval_results.csv",
        help="CSV output path (default: eval_results.csv)",
    )
    p.add_argument(
        "--batch-size", type=int, default=None,
        help="Batch size (defaults to training batch size from config)",
    )
    p.add_argument(
        "--conformal", action="store_true",
        help="Use conformal prediction sets; report coverage/set-size/singleton metrics.",
    )
    p.add_argument(
        "--thresholds-path", default=None,
        help="Path to conformal_thresholds.pt. Defaults to <checkpoint_dir>/conformal_thresholds.pt",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Point-prediction metrics
# ---------------------------------------------------------------------------

def _binary_metrics(preds: np.ndarray, targets: np.ndarray) -> list[dict]:
    """Per-column binary metrics. preds and targets are (N, C) int arrays."""
    rows = []
    for i in range(preds.shape[1]):
        p = preds[:, i]
        t = targets[:, i]
        tp = int(((p == 1) & (t == 1)).sum())
        fp = int(((p == 1) & (t == 0)).sum())
        fn = int(((p == 0) & (t == 1)).sum())
        tn = int(((p == 0) & (t == 0)).sum())
        acc = (tp + tn) / len(t)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        rows.append({
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(t.sum()),
        })
    return rows


def _categorical_metrics(preds: np.ndarray, targets: np.ndarray) -> dict:
    """Multiclass metrics (macro-averaged) for one feature."""
    classes = np.union1d(np.unique(targets), np.unique(preds))
    acc = float((preds == targets).mean())
    precisions, recalls = [], []
    for c in classes:
        tp = int(((preds == c) & (targets == c)).sum())
        fp = int(((preds == c) & (targets != c)).sum())
        fn = int(((preds != c) & (targets == c)).sum())
        precisions.append(tp / (tp + fp) if (tp + fp) > 0 else 0.0)
        recalls.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
    prec = float(np.mean(precisions)) if precisions else 0.0
    rec = float(np.mean(recalls)) if recalls else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "support": int(len(targets))}


# ---------------------------------------------------------------------------
# Conformal metrics
# ---------------------------------------------------------------------------

def _binary_conformal_metrics(
    sets: np.ndarray,    # (N, C, 2) bool — last dim = [include_0, include_1]
    targets: np.ndarray, # (N, C) int
) -> list[dict]:
    """Per-column conformal metrics for binary heads."""
    rows = []
    for i in range(sets.shape[1]):
        s = sets[:, i, :]   # (N, 2)
        t = targets[:, i]   # (N,)
        # covered if true class is in the set
        covered = np.where(t == 1, s[:, 1], s[:, 0])
        set_size = s[:, 0].astype(float) + s[:, 1].astype(float)
        rows.append({
            "coverage": float(covered.mean()),
            "avg_set_size": float(set_size.mean()),
            "singleton_rate": float((set_size == 1).mean()),
            "support": int(t.sum()),
        })
    return rows


def _categorical_conformal_metrics(
    set_mask: np.ndarray,  # (N, K) bool
    targets: np.ndarray,   # (N,) int
) -> dict:
    """Conformal metrics for one categorical feature."""
    n = len(targets)
    covered = set_mask[np.arange(n), targets]
    set_size = set_mask.sum(axis=1).astype(float)
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(set_size.mean()),
        "singleton_rate": float((set_size == 1).mean()),
        "support": n,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_point_results(results: list[dict]) -> None:
    header = (f"{'target':<40} {'type':<12} "
              f"{'acc':>6} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['target']:<40} {r['type']:<12} "
            f"{r['accuracy']:>6.3f} {r['precision']:>6.3f} "
            f"{r['recall']:>6.3f} {r['f1']:>6.3f} {r['support']:>8}"
        )
    for label, type_key in [("binary (MACCS)", "binary"), ("categorical", "categorical")]:
        subset = [r for r in results if r["type"] == type_key]
        if subset:
            print(f"\n{label} macro-avg:")
            for metric in ("accuracy", "precision", "recall", "f1"):
                print(f"  {metric}: {np.mean([r[metric] for r in subset]):.4f}")


def _print_conformal_results(results: list[dict]) -> None:
    header = (f"{'target':<40} {'type':<12} "
              f"{'coverage':>8} {'avg_set':>7} {'singleton':>9} {'support':>8}")
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['target']:<40} {r['type']:<12} "
            f"{r['coverage']:>8.3f} {r['avg_set_size']:>7.3f} "
            f"{r['singleton_rate']:>9.3f} {r['support']:>8}"
        )
    for label, type_key in [("binary (MACCS)", "binary"), ("categorical", "categorical")]:
        subset = [r for r in results if r["type"] == type_key]
        if subset:
            print(f"\n{label} macro-avg:")
            for metric in ("coverage", "avg_set_size", "singleton_rate"):
                print(f"  {metric}: {np.mean([r[metric] for r in subset]):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate arguments
    if args.from_splits and args.test_set:
        print("ERROR: Cannot specify both --test-set and --from-splits", file=sys.stderr)
        sys.exit(1)
    if not args.from_splits and not args.test_set:
        print("ERROR: Must specify either --test-set or --from-splits", file=sys.stderr)
        sys.exit(1)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        config.checkpointing.resolved_save_dir() / "multitask_mlp.best.pt"
    )
    if not checkpoint_path.exists():
        checkpoint_path = config.checkpointing.resolved_save_dir() / "multitask_mlp.latest.pt"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {checkpoint_path}...")
    model, ref_dataset, device = load_model(checkpoint_path)

    # Load test dataset
    if args.from_splits:
        print(f"Loading {args.split_name} split from saved splits...")
        try:
            test_ds, split_info = load_dataset_from_splits(
                split_name=args.split_name,
                split_hash=args.split_hash,
            )
            print(f"Loaded {args.split_name} split from {split_info.get('split_dir', 'saved splits')}")
            print(f"Split info: {split_info}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        test_path = Path(args.test_set)
        if not test_path.exists():
            print(f"Test set not found: {test_path}", file=sys.stderr)
            sys.exit(1)

        schema_path = config.data.resolved_schema_path()
        print(f"Loading test set from JSON: {test_path}")
        
        test_ds = RawFeatureDataset(test_path, schema_path)
        idx_path = test_path.with_name(test_path.name + ".idx")
        print(f"Index file path: {idx_path}")
        print(f"Index file exists: {idx_path.exists()}")
        if idx_path.exists():
            print(f"Index file size: {idx_path.stat().st_size} bytes")
    
    # Apply normalization if available from ref_dataset
    # Handle both Subset and direct dataset objects
    underlying_ds = test_ds.dataset if hasattr(test_ds, 'dataset') else test_ds
    
    if ref_dataset._cont_mean is not None:
        underlying_ds.normalise_continuous(
            ref_dataset._cont_mean.tolist(),
            ref_dataset._cont_std.tolist(),
        )
 
    num_samples = len(test_ds)
    print(f"Loaded dataset with {num_samples} samples.")
    if num_samples == 0:
        print("ERROR: Test dataset is empty! Check that the test set file is not empty and in valid JSON format.")
        sys.exit(1)
    thresholds = None
    if args.conformal:
        thresh_path = (
            Path(args.thresholds_path) if args.thresholds_path
            else checkpoint_path.parent / "conformal_thresholds.pt"
        )
        if not thresh_path.exists():
            print(
                f"Conformal thresholds not found at {thresh_path}. "
                "Run calibrate.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        thresholds = torch.load(thresh_path, map_location=device, weights_only=True)
        print(f"Loaded conformal thresholds from {thresh_path}")

    batch_size = args.batch_size or config.training.batch_size
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Evaluating on {len(test_ds)} samples...")

    num_cat = len(underlying_ds.onehot_feature_order)

    # Accumulators — point mode
    all_binary_preds: list[np.ndarray] = []
    all_binary_targets: list[np.ndarray] = []
    all_cat_preds: list[list[int]] = [[] for _ in range(num_cat)]
    all_cat_targets: list[list[int]] = [[] for _ in range(num_cat)]

    # Accumulators — conformal mode
    all_binary_sets: list[np.ndarray] = []   # (N, C, 2) bool
    all_cat_sets: list[list[np.ndarray]] = [[] for _ in range(num_cat)]  # per feat: (N, K)

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, targets) in enumerate(tqdm(loader, desc="Running inference on test set")):
            x = x.to(device)
            outputs = model(x)

            binary_tgts = targets["binary"].int().numpy()
            all_binary_targets.append(binary_tgts)

            cat_targets_list = targets["categorical"]
            for i, tgt in enumerate(cat_targets_list):
                tgt_i = tgt.numpy() if hasattr(tgt, "numpy") else np.array(tgt)
                all_cat_targets[i].extend(tgt_i.tolist())

            if thresholds is not None:
                pred_sets = sets_from_outputs(outputs, thresholds)
                all_binary_sets.append(pred_sets["binary"].cpu().numpy())
                for i, mask in enumerate(pred_sets["categorical"]):
                    all_cat_sets[i].append(mask.cpu().numpy())
            else:
                binary_probs = torch.sigmoid(outputs["binary"]).cpu()
                binary_preds = (binary_probs >= args.threshold).int().numpy()
                all_binary_preds.append(binary_preds)
                for i, logits in enumerate(outputs["categorical"]):
                    preds_i = logits.argmax(dim=-1).cpu().numpy()
                    all_cat_preds[i].extend(preds_i.tolist())

    binary_targets_arr = np.concatenate(all_binary_targets, axis=0)

    results: list[dict] = []

    if thresholds is not None:
        binary_sets_arr = np.concatenate(all_binary_sets, axis=0)  # (N, C, 2)
        binary_rows = _binary_conformal_metrics(binary_sets_arr, binary_targets_arr)
        for i, metrics in enumerate(binary_rows):
            results.append({"target": f"maccs_{i}", "type": "binary", **metrics})

        for i, feat in enumerate(underlying_ds.onehot_feature_order):
            set_mask = np.concatenate(all_cat_sets[i], axis=0)
            tgts = np.array(all_cat_targets[i])
            metrics = _categorical_conformal_metrics(set_mask, tgts)
            results.append({"target": feat, "type": "categorical", **metrics})

        _print_conformal_results(results)
        fieldnames = ["target", "type", "coverage", "avg_set_size", "singleton_rate", "support"]
    else:
        binary_preds_arr = np.concatenate(all_binary_preds, axis=0)
        binary_rows = _binary_metrics(binary_preds_arr, binary_targets_arr)
        for i, metrics in enumerate(binary_rows):
            results.append({"target": f"maccs_{i}", "type": "binary", **metrics})

        for i, feat in enumerate(underlying_ds.onehot_feature_order):
            metrics = _categorical_metrics(
                np.array(all_cat_preds[i]), np.array(all_cat_targets[i])
            )
            results.append({"target": feat, "type": "categorical", **metrics})

        _print_point_results(results)
        fieldnames = ["target", "type", "accuracy", "precision", "recall", "f1", "support"]

    out_path = Path(args.output)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
