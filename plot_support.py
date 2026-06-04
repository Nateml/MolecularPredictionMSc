#!/usr/bin/env python
"""Plot support distribution for MACCS binary features from a dataset file."""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from config import config
from data.datasets import RawFeatureDataset


def _accumulate_supports(loader: DataLoader) -> tuple[np.ndarray, int]:
    total = torch.zeros(167, dtype=torch.long)
    n = 0
    for _, targets in tqdm(loader, desc="Counting positives"):
        binary = targets["binary"].int()
        total += binary.sum(dim=0)
        n += binary.shape[0]
    print(f"N = {n:,}")
    return total.numpy(), n


def compute_supports(dataset_path: Path, schema_path: Path, batch_size: int = 4096) -> tuple[np.ndarray, int]:
    ds = RawFeatureDataset(dataset_path, schema_path)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    return _accumulate_supports(loader)


def compute_supports_from_subset(subset, batch_size: int = 4096) -> tuple[np.ndarray, int]:
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=4)
    return _accumulate_supports(loader)


def plot(supports: np.ndarray, n_total: int, title_suffix: str, out_path: Path) -> None:
    indices = np.arange(len(supports))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(f"MACCS Key Support Distribution  —  {title_suffix}", fontsize=14, fontweight="bold")

    N = n_total
    norm_s = supports / N
    colors = plt.cm.RdYlGn(norm_s)

    ax1 = axes[0]
    ax1.bar(indices, supports, color=colors, width=0.8, linewidth=0)
    ax1.set_xlabel("MACCS key index")
    ax1.set_ylabel("Support (# positives)")
    ax1.set_title("Per-key support (sorted by index)")
    ax1.set_xlim(-1, len(indices))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.axhline(N * 0.05, color="red", linestyle="--", linewidth=0.8, label="5% N")
    ax1.axhline(N * 0.95, color="blue", linestyle="--", linewidth=0.8, label="95% N")
    ax1.legend(fontsize=9)

    ax2 = axes[1]
    max_s = int(supports.max()) if supports.max() > 0 else 1
    bins = np.logspace(0, np.log10(max_s + 1), 40)
    ax2.hist(supports, bins=bins, color="steelblue", edgecolor="white", linewidth=0.4)
    ax2.set_xscale("log")
    ax2.set_xlabel("Support (log scale)")
    ax2.set_ylabel("# MACCS keys")

    zero_count = int((supports == 0).sum())
    low_count  = int(((supports > 0) & (supports < 1000)).sum())
    mid_count  = int(((supports >= 1000) & (supports < 10000)).sum())
    high_count = int((supports >= 10000).sum())

    ax2.set_title(f"Support histogram  |  zero-support keys: {zero_count}  |  N = {N:,}")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.text(0.98, 0.95,
             f"support=0: {zero_count}\n1–999: {low_count}\n1k–9.9k: {mid_count}\n≥10k: {high_count}",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=9, family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot MACCS key support distribution from a dataset.")
    p.add_argument("dataset", help="Path to dataset JSON file")
    p.add_argument("--schema", default=None, help="Path to schema JSON (default: from config)")
    p.add_argument("--split-pkl", default=None, help="Path to split .pkl (list of indices into dataset)")
    p.add_argument("--output", default=None, help="Output PNG path (default: results/<dataset_stem>_support.png)")
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"ERROR: dataset not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    schema_path = Path(args.schema) if args.schema else config.data.resolved_schema_path()

    if args.split_pkl:
        split_pkl_path = Path(args.split_pkl)
        if not split_pkl_path.exists():
            print(f"ERROR: split pkl not found: {split_pkl_path}", file=sys.stderr)
            sys.exit(1)
        import pickle
        from torch.utils.data import Subset
        full_ds = RawFeatureDataset(dataset_path, schema_path)
        with open(split_pkl_path, "rb") as f:
            indices = pickle.load(f)
        ds_to_use = Subset(full_ds, indices)
        title = f"{dataset_path.name} [{split_pkl_path.stem}]"
        stem = f"{dataset_path.stem}_{split_pkl_path.stem}"
        print(f"Using {len(indices):,} indices from {split_pkl_path.name}")
    else:
        ds_to_use = None
        title = dataset_path.name
        stem = dataset_path.stem

    out_path = Path(args.output) if args.output else Path("results") / f"{stem}_support.png"

    if ds_to_use is not None:
        supports, n_total = compute_supports_from_subset(ds_to_use, args.batch_size)
    else:
        supports, n_total = compute_supports(dataset_path, schema_path, args.batch_size)
    plot(supports, n_total, title, out_path)


if __name__ == "__main__":
    main()
