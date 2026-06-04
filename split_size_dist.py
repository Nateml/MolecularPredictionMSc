#!/usr/bin/env python3
"""
Distribution of molecule sizes in a dataset split.

Usage:
    python split_size_dist.py --dataset data/EI_Dataset_MACCS_Continuous.json \
                              --split data/splits/synthetic-ei/test.pkl

Streams the dataset line-by-line (never loads it fully into memory).
Requires structural_features.heavy_atom_count in each record.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

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


def main() -> None:
    args = parse_args()

    with open(args.split, "rb") as f:
        split_indices: list[int] = pickle.load(f)

    target = set(split_indices)
    print(f"Split: {len(split_indices)} records  (indices {min(split_indices)}–{max(split_indices)})")

    counts: list[int] = []
    n_missing = 0
    rec_idx = 0  # dataset record index (0-based, skipping the opening '[' line)

    with open(args.dataset, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip().rstrip(",")
            if not line or line in ("[", "]"):
                continue

            if rec_idx in target:
                try:
                    rec = json.loads(line)
                    hac = rec.get("structural_features", {}).get("heavy_atom_count")
                    if hac is None:
                        # Fall back to counting from smiles via RDKit if available
                        n_missing += 1
                    else:
                        counts.append(int(hac))
                except json.JSONDecodeError:
                    n_missing += 1

            rec_idx += 1
            # Early exit once we've seen all split indices
            if rec_idx > max(split_indices) + 1:
                break

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
