#!/usr/bin/env python
"""
Evaluate a pretrained checkpoint on a test set.

Computes per-target accuracy, precision, recall, and F1 for all binary
(MACCS key) and categorical (structural feature) heads. Results are printed
to stdout and saved as CSV.

With --conformal, loads conformal_thresholds.pt from the checkpoint directory
and reports coverage rate, average set size, and singleton rate instead.

For models with quantile regression heads, also reports per-feature continuous
metrics (MAE, RMSE, coverage, avg interval size) and atom-count-specific
metrics (median accuracy, joint formula accuracy, coverage, conformal coverage).

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
from model.loader import load_model, load_thresholds
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

ATOM_ELEMENTS = ["element_C", "element_H", "element_N", "element_O", "element_S", "element_P"]
HEAVY_ATOM_ELEMENTS = ["element_C", "element_N", "element_O", "element_S", "element_P"]  # excl. H


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a checkpoint on a test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--test-set", default=None,
        help="Path to test JSON dataset. Mutually exclusive with --from-splits and --split-pkl.",
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
        "--split-pkl", default=None,
        help="Path to a .pkl file containing a list of indices into the full dataset.",
    )
    p.add_argument(
        "--full-dataset", default=None,
        help="Path to the full dataset JSON. Required when --split-pkl is used.",
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
        "--output-folder", default="eval_results/",
        help="path to save CSVs to (default: eval_results/)",
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
        "--alpha", type=float, default=None,
        help="Specifies the significance level of the thresholds to load."
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
    counts = np.bincount(targets)
    support = int(len(targets) - counts.max())
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "support": support}


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
        covered = np.where(t == 1, s[:, 1], s[:, 0])
        set_size = s[:, 0].astype(float) + s[:, 1].astype(float)
        rows.append({
            "coverage": float(covered.mean()),
            "avg_set_size": float(set_size.mean()),
            "set_size_p50": float(np.percentile(set_size, 50)),
            "set_size_p90": float(np.percentile(set_size, 90)),
            "set_size_p95": float(np.percentile(set_size, 95)),
            "efficiency": float(1.0 - set_size.mean() / 2.0),
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
    K = set_mask.shape[1]
    covered = set_mask[np.arange(n), targets]
    set_size = set_mask.sum(axis=1).astype(float)
    return {
        "coverage": float(covered.mean()),
        "avg_set_size": float(set_size.mean()),
        "set_size_p50": float(np.percentile(set_size, 50)),
        "set_size_p90": float(np.percentile(set_size, 90)),
        "set_size_p95": float(np.percentile(set_size, 95)),
        "efficiency": float(1.0 - set_size.mean() / K),
        "singleton_rate": float((set_size == 1).mean()),
        "support": n,
    }


# ---------------------------------------------------------------------------
# Continuous / quantile metrics
# ---------------------------------------------------------------------------

def _unnorm(arr: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    """Invert z-score normalisation. arr last dim must align with mean/std."""
    if mean is None:
        return arr
    if arr.ndim == 3:  # (N, F, Q)
        return arr * std[None, :, None] + mean[None, :, None]
    return arr * std[None, :] + mean[None, :]  # (N, F)


def _continuous_metrics(
    q_orig: np.ndarray,  # (N, F, Q) in original space
    t_orig: np.ndarray,  # (N, F) in original space
    feature_names: list[str],
    conf_lo: np.ndarray | None = None,  # (N, F)
    conf_hi: np.ndarray | None = None,  # (N, F)
) -> list[dict]:
    """Per-feature continuous metrics."""
    N, F, Q = q_orig.shape
    mid_idx = Q // 2
    lo = q_orig[:, :, 0]      # (N, F)
    med = q_orig[:, :, mid_idx]
    hi = q_orig[:, :, -1]

    rows = []
    for f in range(F):
        mae = float(np.abs(med[:, f] - t_orig[:, f]).mean())
        rmse = float(np.sqrt(((med[:, f] - t_orig[:, f]) ** 2).mean()))
        covered = (lo[:, f] <= t_orig[:, f]) & (t_orig[:, f] <= hi[:, f])
        coverage = float(covered.mean())
        avg_interval = float((hi[:, f] - lo[:, f]).mean())
        row: dict = {
            "feature": feature_names[f],
            "mae": mae,
            "rmse": rmse,
            "coverage": coverage,
            "avg_interval_size": avg_interval,
        }
        if conf_lo is not None and conf_hi is not None:
            c_covered = (conf_lo[:, f] <= t_orig[:, f]) & (t_orig[:, f] <= conf_hi[:, f])
            c_size = conf_hi[:, f] - conf_lo[:, f]
            row["conformal_coverage"] = float(c_covered.mean())
            w_sum = float(c_size.sum())
            row["conformal_coverage_weighted"] = (
                float((c_covered.astype(float) * c_size).sum() / w_sum)
                if w_sum > 0 else float("nan")
            )
            row["conformal_avg_interval_size"] = float(c_size.mean())
        rows.append(row)
    return rows


def _atom_count_metrics(
    q_orig: np.ndarray,       # (N, F, Q)
    t_orig: np.ndarray,       # (N, F)
    feature_names: list[str],
    conf_lo: np.ndarray | None = None,  # (N, F)
    conf_hi: np.ndarray | None = None,
) -> tuple[list[dict], float, float | None, float | None]:
    """
    Atom-count-specific metrics for element_* features.

    Returns:
        atom_rows: per-element dict list
        joint_accuracy: fraction where all median preds match true counts
        joint_conformal_coverage: fraction where all conformal intervals cover truth (or None)
        joint_conformal_coverage_weighted: weighted version (or None)
    """
    N, F, Q = q_orig.shape
    mid_idx = Q // 2
    lo = q_orig[:, :, 0]
    med = q_orig[:, :, mid_idx]
    hi = q_orig[:, :, -1]

    atom_indices = [
        (elem, feature_names.index(elem))
        for elem in ATOM_ELEMENTS
        if elem in feature_names
    ]

    atom_rows = []
    joint_correct = np.ones(N, dtype=bool)
    joint_conf_covered = np.ones(N, dtype=bool) if conf_lo is not None else None

    for elem, fi in atom_indices:
        med_int = np.round(med[:, fi]).astype(int)
        true_int = np.round(t_orig[:, fi]).astype(int)
        correct = med_int == true_int

        raw_covered = (lo[:, fi] <= t_orig[:, fi]) & (t_orig[:, fi] <= hi[:, fi])
        interval_size = hi[:, fi] - lo[:, fi]

        row: dict = {
            "element": elem,
            "median_accuracy": float(correct.mean()),
            "coverage": float(raw_covered.mean()),
            "avg_interval_size": float(interval_size.mean()),
        }

        joint_correct &= correct

        if conf_lo is not None and conf_hi is not None:
            c_cov = (conf_lo[:, fi] <= t_orig[:, fi]) & (t_orig[:, fi] <= conf_hi[:, fi])
            c_size = conf_hi[:, fi] - conf_lo[:, fi]
            c_sum = float(c_size.sum())
            row["conformal_coverage"] = float(c_cov.mean())
            row["conformal_coverage_weighted"] = (
                float((c_cov.astype(float) * c_size).sum() / c_sum)
                if c_sum > 0 else float("nan")
            )
            row["conformal_avg_interval_size"] = float(c_size.mean())
            if joint_conf_covered is not None:
                joint_conf_covered &= c_cov

        atom_rows.append(row)

    joint_acc = float(joint_correct.mean()) if atom_indices else float("nan")

    joint_conf_cov = None
    joint_conf_cov_w = None
    if joint_conf_covered is not None and atom_indices:
        # Weight by sum of all atom interval sizes per molecule
        conf_atom_sizes = np.stack(
            [conf_hi[:, fi] - conf_lo[:, fi] for _, fi in atom_indices], axis=1
        ).sum(axis=1)
        joint_conf_cov = float(joint_conf_covered.mean())
        w_sum = float(conf_atom_sizes.sum())
        joint_conf_cov_w = (
            float((joint_conf_covered.astype(float) * conf_atom_sizes).sum() / w_sum)
            if w_sum > 0 else float("nan")
        )

    # Formula count: integer combinations in atom count intervals
    def _icount(lo_arr: np.ndarray, hi_arr: np.ndarray, fi: int) -> np.ndarray:
        return np.maximum(1.0, np.floor(hi_arr[:, fi]) - np.ceil(lo_arr[:, fi]) + 1)

    fc_stats_raw: dict | None = None
    if atom_indices:
        raw_fc = np.ones(N, dtype=float)
        for _, fi in atom_indices:
            raw_fc *= _icount(lo, hi, fi)
        fc_stats_raw = {
            "mean": float(raw_fc.mean()),
            "median": float(np.median(raw_fc)),
            "p90": float(np.percentile(raw_fc, 90)),
            "p95": float(np.percentile(raw_fc, 95)),
        }

    fc_stats_conf: dict | None = None
    if conf_lo is not None and conf_hi is not None and atom_indices:
        conf_fc = np.ones(N, dtype=float)
        for _, fi in atom_indices:
            conf_fc *= _icount(conf_lo, conf_hi, fi)
        fc_stats_conf = {
            "mean": float(conf_fc.mean()),
            "median": float(np.median(conf_fc)),
            "p90": float(np.percentile(conf_fc, 90)),
            "p95": float(np.percentile(conf_fc, 95)),
        }

    return atom_rows, joint_acc, joint_conf_cov, joint_conf_cov_w, fc_stats_raw, fc_stats_conf


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_point_results(results: list[dict], n_total: int = 0) -> None:
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
            included = [r for r in subset if 0 < r["support"] < n_total]
            n_excluded = len(subset) - len(included)
            note = f"excluding {n_excluded} degenerate classes (support == 0 or == N)"
            print(f"\n{label} macro-avg ({note}):")
            for metric in ("accuracy", "precision", "recall", "f1"):
                vals = [r[metric] for r in included]
                print(f"  {metric}: {np.mean(vals):.4f}" if vals else f"  {metric}: N/A")


def _print_conformal_results(results: list[dict], n_total: int = 0) -> None:
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
            included = [r for r in subset if 0 < r["support"] < n_total]
            n_excluded = len(subset) - len(included)
            note = f"excluding {n_excluded} degenerate classes (support == 0 or == N)"
            print(f"\n{label} macro-avg ({note}):")
            for metric in ("coverage", "avg_set_size", "set_size_p50", "set_size_p90",
                           "set_size_p95", "efficiency", "singleton_rate"):
                vals = [r[metric] for r in included if metric in r]
                print(f"  {metric}: {np.mean(vals):.4f}" if vals else f"  {metric}: N/A")


def _print_continuous_results(
    cont_rows: list[dict],
    atom_rows: list[dict],
    joint_acc: float,
    joint_conf_cov: float | None,
    joint_conf_cov_w: float | None,
    has_conformal: bool,
) -> None:
    # All continuous features table
    print("\n--- Continuous feature metrics ---")
    if has_conformal:
        hdr = (f"{'feature':<40} {'mae':>8} {'rmse':>8} {'cov':>6} "
               f"{'interval':>10} {'conf_cov':>9} {'conf_cov_w':>11} {'conf_ivl':>9}")
    else:
        hdr = (f"{'feature':<40} {'mae':>8} {'rmse':>8} {'cov':>6} {'interval':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in cont_rows:
        if has_conformal:
            print(
                f"{r['feature']:<40} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
                f"{r['coverage']:>6.3f} {r['avg_interval_size']:>10.3f} "
                f"{r.get('conformal_coverage', float('nan')):>9.3f} "
                f"{r.get('conformal_coverage_weighted', float('nan')):>11.3f} "
                f"{r.get('conformal_avg_interval_size', float('nan')):>9.3f}"
            )
        else:
            print(
                f"{r['feature']:<40} {r['mae']:>8.4f} {r['rmse']:>8.4f} "
                f"{r['coverage']:>6.3f} {r['avg_interval_size']:>10.3f}"
            )
    macro_mae = np.mean([r["mae"] for r in cont_rows])
    macro_cov = np.mean([r["coverage"] for r in cont_rows])
    print(f"\ncontinuous macro-avg: mae={macro_mae:.4f}  coverage={macro_cov:.4f}")

    # Atom count table
    if not atom_rows:
        return
    print("\n--- Atom count metrics ---")
    if has_conformal:
        hdr2 = (f"{'element':<14} {'med_acc':>8} {'cov':>6} {'interval':>10} "
                f"{'conf_cov':>9} {'conf_cov_w':>11} {'conf_ivl':>9}")
    else:
        hdr2 = f"{'element':<14} {'med_acc':>8} {'cov':>6} {'interval':>10}"
    print(hdr2)
    print("-" * len(hdr2))
    for r in atom_rows:
        if has_conformal:
            print(
                f"{r['element']:<14} {r['median_accuracy']:>8.3f} "
                f"{r['coverage']:>6.3f} {r['avg_interval_size']:>10.3f} "
                f"{r.get('conformal_coverage', float('nan')):>9.3f} "
                f"{r.get('conformal_coverage_weighted', float('nan')):>11.3f} "
                f"{r.get('conformal_avg_interval_size', float('nan')):>9.3f}"
            )
        else:
            print(
                f"{r['element']:<14} {r['median_accuracy']:>8.3f} "
                f"{r['coverage']:>6.3f} {r['avg_interval_size']:>10.3f}"
            )
    print(f"\njoint formula accuracy (all atoms correct): {joint_acc:.4f}")
    if has_conformal and joint_conf_cov is not None:
        print(f"joint conformal coverage:                   {joint_conf_cov:.4f}")
        print(f"joint conformal coverage (weighted):        {joint_conf_cov_w:.4f}")


def _print_formula_count_stats(fc_raw: dict | None, fc_conf: dict | None) -> None:
    if fc_raw is None and fc_conf is None:
        return
    print("\n--- Formula count (cross-product of atom count intervals) ---")
    hdr = f"{'source':<16} {'mean':>10} {'median':>10} {'p90':>10} {'p95':>10}"
    print(hdr)
    print("-" * len(hdr))
    for label, stats in [("raw quantile", fc_raw), ("conformal", fc_conf)]:
        if stats is not None:
            print(
                f"{label:<16} {stats['mean']:>10.1f} {stats['median']:>10.1f}"
                f" {stats['p90']:>10.1f} {stats['p95']:>10.1f}"
            )


def _stratified_coverage_by_size(
    t_orig: np.ndarray,
    feat_names: list[str],
    conf_lo_orig: np.ndarray | None,
    conf_hi_orig: np.ndarray | None,
    bin_covered_mat: np.ndarray | None,
    cat_covered_mat: np.ndarray | None,
) -> list[dict]:
    """Joint conformal coverage stratified by heavy atom count (C+N+O+S+P)."""
    heavy_fi = [feat_names.index(e) for e in HEAVY_ATOM_ELEMENTS if e in feat_names]
    if not heavy_fi:
        return []
    heavy_count = np.round(t_orig[:, heavy_fi]).astype(int).sum(axis=1)

    atom_fi = [feat_names.index(e) for e in ATOM_ELEMENTS if e in feat_names]
    buckets = [("<=10", 0, 10), ("11-18", 11, 18), ("19-26", 19, 26), (">26", 27, 9_999)]
    rows = []
    for label, lo_b, hi_b in buckets:
        mask = (heavy_count >= lo_b) & (heavy_count <= hi_b)
        n = int(mask.sum())
        if n == 0:
            continue
        row: dict = {"bucket": label, "n": n}
        if conf_lo_orig is not None and conf_hi_orig is not None:
            cont_cov = (
                (conf_lo_orig[mask] <= t_orig[mask]) & (t_orig[mask] <= conf_hi_orig[mask])
            )
            if atom_fi:
                row["atom_joint_conf_cov"] = float(cont_cov[:, atom_fi].all(axis=1).mean())
            row["cont_joint_conf_cov"] = float(cont_cov.all(axis=1).mean())
        if bin_covered_mat is not None:
            row["binary_joint_conf_cov"] = float(bin_covered_mat[mask].all(axis=1).mean())
        if cat_covered_mat is not None:
            row["cat_joint_conf_cov"] = float(cat_covered_mat[mask].all(axis=1).mean())
        rows.append(row)
    return rows


def _per_element_stratified_coverage(
    t_orig: np.ndarray,
    feat_names: list[str],
    conf_lo_orig: np.ndarray,
    conf_hi_orig: np.ndarray,
) -> list[dict]:
    """Per-element conformal coverage stratified by true count magnitude."""
    atom_indices = [(e, feat_names.index(e)) for e in ATOM_ELEMENTS if e in feat_names]
    buckets = [("0", 0, 0), ("1-3", 1, 3), ("4-7", 4, 7), ("8-12", 8, 12), ("13+", 13, 9_999)]
    rows = []
    for elem, fi in atom_indices:
        true_int = np.round(t_orig[:, fi]).astype(int)
        c_cov = (conf_lo_orig[:, fi] <= t_orig[:, fi]) & (t_orig[:, fi] <= conf_hi_orig[:, fi])
        for blabel, lo_b, hi_b in buckets:
            bmask = (true_int >= lo_b) & (true_int <= hi_b)
            n = int(bmask.sum())
            if n == 0:
                continue
            rows.append({
                "element": elem,
                "bucket": blabel,
                "n": n,
                "conformal_coverage": float(c_cov[bmask].mean()),
            })
    return rows


def _print_stratified_coverage(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n--- Joint conformal coverage by heavy atom count ---")
    metric_keys = [k for k in rows[0] if k not in ("bucket", "n")]
    hdr = f"{'bucket':<8} {'n':>6}" + "".join(f" {k:>26}" for k in metric_keys)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['bucket']:<8} {r['n']:>6}"
        for k in metric_keys:
            v = r.get(k)
            line += f" {v:>26.4f}" if v is not None else f" {'N/A':>26}"
        print(line)


def _print_per_element_stratified(rows: list[dict]) -> None:
    if not rows:
        return
    print("\n--- Per-element conformal coverage by true count ---")
    hdr = f"{'element':<14} {'bucket':<8} {'n':>6} {'conf_cov':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['element']:<14} {r['bucket']:<8} {r['n']:>6} {r['conformal_coverage']:>10.4f}")


def _print_joint_summary(
    joint_binary_conf_cov: float | None,
    joint_cat_conf_cov: float | None,
    joint_cont_conf_cov: float | None,
    joint_atom_conf_cov: float | None,
    joint_atom_conf_cov_w: float | None,
    joint_all_excl_atoms_conf_cov: float | None,
) -> None:
    items = [
        ("binary (all MACCS):",               joint_binary_conf_cov),
        ("categorical (all):",                joint_cat_conf_cov),
        ("continuous (all features):",        joint_cont_conf_cov),
        ("atom counts only:",                 joint_atom_conf_cov),
        ("atom counts only (weighted):",      joint_atom_conf_cov_w),
        ("all features (excl. atom counts):", joint_all_excl_atoms_conf_cov),
    ]
    if all(v is None for _, v in items):
        return
    print("\n--- Joint conformal coverage summary ---")
    for label, val in items:
        if val is not None:
            print(f"  {label:<42} {val:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Validate arguments
    modes = sum([bool(args.test_set), args.from_splits, bool(args.split_pkl)])
    if modes > 1:
        print("ERROR: --test-set, --from-splits, and --split-pkl are mutually exclusive", file=sys.stderr)
        sys.exit(1)
    if modes == 0:
        print("ERROR: Must specify one of --test-set, --from-splits, or --split-pkl", file=sys.stderr)
        sys.exit(1)
    if args.split_pkl and not args.full_dataset:
        print("ERROR: --full-dataset is required when using --split-pkl", file=sys.stderr)
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
        _schema_path = config.data.resolved_schema_path()
        _dataset_str = str(config.data.resolved_dataset_path().resolve())
        _split_str = f"{args.split_name} (hash: {split_info.get('split_hash', args.split_hash or 'computed')})"
    elif args.split_pkl:
        split_pkl_path = Path(args.split_pkl)
        full_ds_path = Path(args.full_dataset)
        if not split_pkl_path.exists():
            print(f"Split pkl not found: {split_pkl_path}", file=sys.stderr)
            sys.exit(1)
        if not full_ds_path.exists():
            print(f"Full dataset not found: {full_ds_path}", file=sys.stderr)
            sys.exit(1)
        schema_path = config.data.resolved_schema_path()
        print(f"Loading full dataset from {full_ds_path}...")
        full_ds = RawFeatureDataset(full_ds_path, schema_path)
        with open(split_pkl_path, "rb") as f:
            indices = pickle.load(f)
        from torch.utils.data import Subset
        test_ds = Subset(full_ds, indices)
        print(f"Loaded {len(indices)} indices from {split_pkl_path}")
        _schema_path = schema_path
        _dataset_str = str(full_ds_path.resolve())
        _split_str = f"pkl: {split_pkl_path.resolve()}"
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
        _schema_path = schema_path
        _dataset_str = str(test_path.resolve())
        _split_str = "N/A (full JSON)"
    
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
        thresh_path = args.thresholds_path
        if not thresh_path:
            thresholds = load_thresholds(checkpoint_path, alpha=args.alpha if args.alpha else config.conformal.alpha)
            print("Found conformal thresholds using checkpoint path and significance level.")
        else:
            thresholds = torch.load(thresh_path, map_location='cpu')
            print(f"Loaded conformal thresholds from {thresh_path}")

    batch_size = args.batch_size or config.training.batch_size
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    print(f"Evaluating on {len(test_ds)} samples...")

    num_cat = len(underlying_ds.onehot_feature_order)
    has_quantile = getattr(model, "num_regression", 0) > 0

    # Accumulators — point mode
    all_binary_preds: list[np.ndarray] = []
    all_binary_targets: list[np.ndarray] = []
    all_cat_preds: list[list[int]] = [[] for _ in range(num_cat)]
    all_cat_targets: list[list[int]] = [[] for _ in range(num_cat)]

    # Accumulators — conformal mode
    all_binary_sets: list[np.ndarray] = []
    all_cat_sets: list[list[np.ndarray]] = [[] for _ in range(num_cat)]

    # Accumulators — continuous/quantile
    all_quantile_preds: list[np.ndarray] = []   # (B, F, Q) each
    all_cont_targets: list[np.ndarray] = []     # (B, F) each
    all_conf_quantile_lo: list[np.ndarray] = [] # (B, F)
    all_conf_quantile_hi: list[np.ndarray] = []

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

            # Continuous targets
            if has_quantile and "continuous" in targets:
                cont_t = targets["continuous"].float().numpy()  # (B, F)
                all_cont_targets.append(cont_t)

            if thresholds is not None:
                pred_sets = sets_from_outputs(outputs, thresholds, quantile_high_idx=getattr(model, "num_quantiles", 5) - 1)
                all_binary_sets.append(pred_sets["binary"].cpu().numpy())
                for i, mask in enumerate(pred_sets["categorical"]):
                    all_cat_sets[i].append(mask.cpu().numpy())
                if has_quantile and pred_sets.get("quantile") is not None:
                    q_sets = pred_sets["quantile"].cpu().numpy()  # (B, F, 2)
                    all_conf_quantile_lo.append(q_sets[:, :, 0])
                    all_conf_quantile_hi.append(q_sets[:, :, 1])
            else:
                binary_probs = torch.sigmoid(outputs["binary"]).cpu()
                binary_preds = (binary_probs >= args.threshold).int().numpy()
                all_binary_preds.append(binary_preds)
                for i, logits in enumerate(outputs["categorical"]):
                    preds_i = logits.argmax(dim=-1).cpu().numpy()
                    all_cat_preds[i].extend(preds_i.tolist())

            # Collect quantile preds (both modes)
            if has_quantile and "quantile" in outputs and outputs["quantile"]:
                q_stack = torch.stack(outputs["quantile"], dim=1).cpu().numpy()  # (B, F, Q)
                all_quantile_preds.append(q_stack)

    binary_targets_arr = np.concatenate(all_binary_targets, axis=0)
    n_total = binary_targets_arr.shape[0]

    results: list[dict] = []

    joint_binary_conf_cov: float | None = None
    joint_cat_conf_cov: float | None = None
    joint_cont_conf_cov: float | None = None
    joint_all_excl_atoms_conf_cov: float | None = None
    _bin_covered_mat: np.ndarray | None = None
    _cat_covered_mat: np.ndarray | None = None

    import io

    class _Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, data):
            for f in self.files:
                f.write(data)
        def flush(self):
            for f in self.files:
                f.flush()

    import datetime
    _run_ts = datetime.datetime.now()
    _run_info = {
        "timestamp":  _run_ts.isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint_path.resolve()),
        "schema":     str(_schema_path.resolve()),
        "dataset":    _dataset_str,
        "split":      _split_str,
        "n_samples":  num_samples,
        "mode":       "conformal" if thresholds is not None else "point",
    }
    if args.conformal:
        _run_info["alpha"] = args.alpha if args.alpha else config.conformal.alpha

    txt_buffer = io.StringIO()
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, txt_buffer)

    print("=== Evaluation run ===")
    for _k, _v in _run_info.items():
        print(f"  {_k:<12} {_v}")
    print("=" * 22)

    if thresholds is not None:
        binary_sets_arr = np.concatenate(all_binary_sets, axis=0)
        binary_rows = _binary_conformal_metrics(binary_sets_arr, binary_targets_arr)
        for i, metrics in enumerate(binary_rows):
            results.append({"target": f"maccs_{i}", "type": "binary", **metrics})

        _bin_covered_mat = np.where(
            binary_targets_arr == 1, binary_sets_arr[:, :, 1], binary_sets_arr[:, :, 0]
        )  # (N, C)
        joint_binary_conf_cov = float(_bin_covered_mat.all(axis=1).mean())

        _cat_covered_cols: list[np.ndarray] = []
        for i, feat in enumerate(underlying_ds.onehot_feature_order):
            set_mask = np.concatenate(all_cat_sets[i], axis=0)
            tgts = np.array(all_cat_targets[i])
            metrics = _categorical_conformal_metrics(set_mask, tgts)
            results.append({"target": feat, "type": "categorical", **metrics})
            _cat_covered_cols.append(set_mask[np.arange(n_total), tgts])

        if _cat_covered_cols:
            _cat_covered_mat = np.stack(_cat_covered_cols, axis=1)  # (N, num_cat)
            joint_cat_conf_cov = float(_cat_covered_mat.all(axis=1).mean())

        _print_conformal_results(results, n_total=n_total)
        fieldnames = ["target", "type", "coverage", "avg_set_size",
                      "set_size_p50", "set_size_p90", "set_size_p95",
                      "efficiency", "singleton_rate", "support"]
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

        _print_point_results(results, n_total=n_total)
        fieldnames = ["target", "type", "accuracy", "precision", "recall", "f1", "support"]

    # ---------------------------------------------------------------------------
    # Continuous metrics
    # ---------------------------------------------------------------------------
    cont_rows: list[dict] = []
    atom_rows: list[dict] = []
    joint_acc = float("nan")
    joint_conf_cov = None
    joint_conf_cov_w = None
    fc_stats_raw: dict | None = None
    fc_stats_conf: dict | None = None
    strat_rows: list[dict] = []
    elem_strat_rows: list[dict] = []

    if all_quantile_preds and all_cont_targets:
        q_arr = np.concatenate(all_quantile_preds, axis=0)  # (N, F, Q)
        t_arr = np.concatenate(all_cont_targets, axis=0)    # (N, F)

        cont_mean = underlying_ds._cont_mean.numpy() if underlying_ds._cont_mean is not None else None
        cont_std = underlying_ds._cont_std.numpy() if underlying_ds._cont_std is not None else None

        q_orig = _unnorm(q_arr, cont_mean, cont_std)  # (N, F, Q)
        t_orig = _unnorm(t_arr, cont_mean, cont_std)  # (N, F)

        conf_lo_orig = conf_hi_orig = None
        if all_conf_quantile_lo and all_conf_quantile_hi:
            lo_norm = np.concatenate(all_conf_quantile_lo, axis=0)  # (N, F)
            hi_norm = np.concatenate(all_conf_quantile_hi, axis=0)
            conf_lo_orig = _unnorm(lo_norm, cont_mean, cont_std)
            conf_hi_orig = _unnorm(hi_norm, cont_mean, cont_std)

        feat_names = underlying_ds.continuous_feature_order
        cont_rows = _continuous_metrics(q_orig, t_orig, feat_names, conf_lo_orig, conf_hi_orig)
        atom_rows, joint_acc, joint_conf_cov, joint_conf_cov_w, fc_stats_raw, fc_stats_conf = (
            _atom_count_metrics(q_orig, t_orig, feat_names, conf_lo_orig, conf_hi_orig)
        )

        _print_continuous_results(
            cont_rows, atom_rows, joint_acc, joint_conf_cov, joint_conf_cov_w,
            has_conformal=conf_lo_orig is not None,
        )
        _print_formula_count_stats(fc_stats_raw, fc_stats_conf)

        if conf_lo_orig is not None and conf_hi_orig is not None:
            cont_all_covered = (conf_lo_orig <= t_orig) & (t_orig <= conf_hi_orig)  # (N, F)
            joint_cont_conf_cov = float(cont_all_covered.all(axis=1).mean())

            atom_fi_set = {feat_names.index(e) for e in ATOM_ELEMENTS if e in feat_names}
            non_atom_idx = [i for i in range(len(feat_names)) if i not in atom_fi_set]
            excl_parts: list[np.ndarray] = [cont_all_covered[:, non_atom_idx]]
            if _bin_covered_mat is not None:
                excl_parts.append(_bin_covered_mat)
            if _cat_covered_mat is not None:
                excl_parts.append(_cat_covered_mat)
            joint_all_excl_atoms_conf_cov = float(
                np.concatenate(excl_parts, axis=1).all(axis=1).mean()
            )

            strat_rows = _stratified_coverage_by_size(
                t_orig, feat_names, conf_lo_orig, conf_hi_orig,
                _bin_covered_mat, _cat_covered_mat,
            )
            _print_stratified_coverage(strat_rows)

            elem_strat_rows = _per_element_stratified_coverage(
                t_orig, feat_names, conf_lo_orig, conf_hi_orig
            )
            _print_per_element_stratified(elem_strat_rows)

    _print_joint_summary(
        joint_binary_conf_cov, joint_cat_conf_cov,
        joint_cont_conf_cov, joint_conf_cov, joint_conf_cov_w,
        joint_all_excl_atoms_conf_cov,
    )

    sys.stdout = _orig_stdout

    # ---------------------------------------------------------------------------
    # Save CSVs
    # ---------------------------------------------------------------------------
    ts = _run_ts.strftime("%Y%m%d_%H%M%S")

    out_dir = Path(args.output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    txt_out = out_dir / f"eval_results_{ts}.txt"
    with open(txt_out, "w") as f:
        f.write(txt_buffer.getvalue())
    print(f"\nText results saved to {txt_out}")

    main_out = out_dir / f"eval_results_{ts}.csv"
    with open(main_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {main_out}")

    if cont_rows:
        cont_out = out_dir / f"eval_results_continuous_{ts}.csv"
        cont_fieldnames = list(cont_rows[0].keys())
        with open(cont_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cont_fieldnames)
            writer.writeheader()
            writer.writerows(cont_rows)
        print(f"Continuous results saved to {cont_out}")

    if atom_rows:
        atom_out = out_dir / f"eval_results_atom_counts_{ts}.csv"
        atom_fieldnames = list(atom_rows[0].keys())
        with open(atom_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=atom_fieldnames)
            writer.writeheader()
            writer.writerows(atom_rows)
            writer.writerow({k: "" for k in atom_fieldnames})
            summary: dict = {
                "element": "JOINT",
                "median_accuracy": joint_acc,
                "coverage": float("nan"),
                "avg_interval_size": float("nan"),
            }
            if "conformal_coverage" in atom_fieldnames:
                summary["conformal_coverage"] = joint_conf_cov if joint_conf_cov is not None else float("nan")
                summary["conformal_coverage_weighted"] = joint_conf_cov_w if joint_conf_cov_w is not None else float("nan")
                summary["conformal_avg_interval_size"] = float("nan")
            writer.writerow(summary)
        print(f"Atom count results saved to {atom_out}")

    if strat_rows:
        strat_out = out_dir / f"eval_results_stratified_{ts}.csv"
        with open(strat_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(strat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(strat_rows)
        print(f"Stratified coverage saved to {strat_out}")

    if elem_strat_rows:
        elem_strat_out = out_dir / f"eval_results_elem_stratified_{ts}.csv"
        with open(elem_strat_out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(elem_strat_rows[0].keys()))
            writer.writeheader()
            writer.writerows(elem_strat_rows)
        print(f"Per-element stratified coverage saved to {elem_strat_out}")


if __name__ == "__main__":
    main()
