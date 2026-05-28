#!/usr/bin/env python
"""
Evaluate a pretrained checkpoint on a test set.

Computes per-target accuracy, precision, recall, and F1 for all binary
(MACCS key) and categorical (structural feature) heads. Results are printed
to stdout and saved as CSV.

With --conformal, loads conformal_thresholds.pt from the checkpoint directory
and reports coverage rate, average set size, and singleton rate instead.

Usage:
    python evaluate.py --test-set data/test.json
    python evaluate.py --test-set data/test.json --checkpoint models/final/multitask_mlp_mixed_best.pt
    python evaluate.py --test-set data/test.json --threshold 0.5 --output results.csv
    python evaluate.py --test-set data/test.json --conformal
    python evaluate.py --test-set data/test.json --conformal --thresholds-path path/to/conformal_thresholds.pt
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

from config import config
from data.datasets import RawFeatureDataset
from model.loader import load_model
from model.conformal import sets_from_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--test-set", required=True, help="Path to test JSON dataset.")
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

    test_path = Path(args.test_set)
    if not test_path.exists():
        print(f"Test set not found: {test_path}", file=sys.stderr)
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

    schema_path = config.data.resolved_schema_path()
    test_ds = RawFeatureDataset(test_path, schema_path)
    if ref_dataset._cont_mean is not None:
        test_ds.normalise_continuous(
            ref_dataset._cont_mean.tolist(),
            ref_dataset._cont_std.tolist(),
        )

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

    num_cat = len(test_ds.onehot_feature_order)

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
            if batch_idx >= 2:
                break
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

        for i, feat in enumerate(test_ds.onehot_feature_order):
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

        for i, feat in enumerate(test_ds.onehot_feature_order):
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
