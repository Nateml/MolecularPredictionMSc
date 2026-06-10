#!/usr/bin/env python3
"""
Distribution of molecule sizes in a dataset split.

Usage:
    python split_size_dist.py --dataset data/multitask_mlp_hardcoded.test.json \
                              --split data/splits/f9d711f95583/test.pkl

Uses the dataset index file (<dataset>.idx) to seek directly to each record
instead of scanning the full dataset line-by-line.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from tqdm import tqdm

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to full dataset .json file.")
    p.add_argument("--split", required=True, help="Path to split indices .pkl file.")
    p.add_argument(
        "--buckets", default="0,10,18,26",
        help="Comma-separated heavy-atom-count bucket edges (default: 0,10,18,26).",
    )
    return p.parse_args()


def load_index(dataset_path: Path) -> list[tuple[int, int]]:
    idx_path = dataset_path.with_name(dataset_path.name + ".idx")
    if not idx_path.exists():
        raise FileNotFoundError(
            f"Index file not found: {idx_path}\n"
            "Run the dataset loader once to generate it."
        )
    with open(idx_path, "r", encoding="utf-8") as fh:
        return [tuple(pair) for pair in json.load(fh)]


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)

    with open(args.split, "rb") as f:
        split_indices: list[int] = pickle.load(f)

    print(f"Split: {len(split_indices)} records  (indices {min(split_indices)}–{max(split_indices)})")

    index = load_index(dataset_path)

    counts: list[int] = []
    n_missing = 0

    with open(dataset_path, "rb") as f:
        for idx in tqdm(split_indices):
            start, end = index[idx]
            f.seek(start)
            raw = f.read(end - start)
            try:
                rec = json.loads(raw)
                hac = rec.get("structural_features", {}).get("heavy_atom_count")
                if hac is None:
                    n_missing += 1
                else:
                    counts.append(int(hac))
            except json.JSONDecodeError:
                n_missing += 1

    if not counts:
        print("No heavy_atom_count values found.")
        return

    arr = np.array(counts, dtype=int)

    print(f"\nRecords with heavy_atom_count : {len(counts)}")
    if n_missing:
        print(f"Records missing field         : {n_missing}")

    print(f"\n--- Summary ---")
    print(f"  min    : {arr.min()}")
    print(f"  p25    : {int(np.percentile(arr, 25))}")
    print(f"  median : {int(np.median(arr))}")
    print(f"  p75    : {int(np.percentile(arr, 75))}")
    print(f"  p90    : {int(np.percentile(arr, 90))}")
    print(f"  p95    : {int(np.percentile(arr, 95))}")
    print(f"  max    : {arr.max()}")
    print(f"  mean   : {arr.mean():.2f}")
    print(f"  std    : {arr.std():.2f}")

    edges = [int(e) for e in args.buckets.split(",")]
    bucket_labels = [f"<={edges[0]}"] + [f"{edges[i]+1}-{edges[i+1]}" for i in range(len(edges)-1)] + [f">{edges[-1]}"]
    bucket_counts = [0] * len(bucket_labels)

    for v in counts:
        if v <= edges[0]:
            bucket_counts[0] += 1
        elif v > edges[-1]:
            bucket_counts[-1] += 1
        else:
            for i in range(len(edges) - 1):
                if edges[i] < v <= edges[i + 1]:
                    bucket_counts[i + 1] += 1
                    break

    print(f"\n--- Buckets (heavy atom count) ---")
    total = len(counts)
    for label, cnt in zip(bucket_labels, bucket_counts):
        bar = "#" * int(50 * cnt / total)
        print(f"  {label:<10} {cnt:>7}  ({100*cnt/total:5.1f}%)  {bar}")

    print(f"\n--- Top 20 most common sizes ---")
    for size, freq in sorted(Counter(counts).items(), key=lambda x: -x[1])[:20]:
        bar = "#" * int(40 * freq / Counter(counts).most_common(1)[0][1])
        print(f"  {size:>3} atoms : {freq:>6}  ({100*freq/total:4.1f}%)  {bar}")


if __name__ == "__main__":
    main()
