#!/usr/bin/env python3
"""Load the continuous-feature MoNA dataset and print a short summary.

Usage:
  python scripts/load_continuous_dataset.py --data data/MoNA-filtered-curated-with-maccs-and-structure-deduped.json \
      --schema data/feature_vector_continuous.json
"""

from pathlib import Path
import argparse
import json

# Import as package-local module
from data.datasets import RawFeatureDataset


def main() -> None:
    p = argparse.ArgumentParser(description="Load continuous-feature dataset and show summary")
    p.add_argument(
        "--data",
        default="data/MoNA-filtered-curated-with-maccs-and-structure-deduped.json",
        help="path to the JSON dataset file",
    )
    p.add_argument(
        "--schema",
        default="data/feature_vector_continuous.json",
        help="path to the continuous feature schema JSON",
    )
    args = p.parse_args()

    data_path = Path(args.data)
    schema_path = Path(args.schema)

    print(f"Loading dataset: {data_path} with schema: {schema_path}")
    ds = RawFeatureDataset(data_path, schema_path)

    # If an existing .idx file is present but empty, rebuild the index.
    idx_path = data_path.with_name(data_path.name + ".idx")
    if idx_path.exists():
        try:
            with open(idx_path, "r", encoding="utf-8") as fh:
                idx_contents = fh.read().strip()
        except Exception:
            idx_contents = ""
        if not idx_contents or idx_contents == "[]":
            print(f"Index file {idx_path} is empty — rebuilding index")
            try:
                idx_path.unlink()
            except Exception:
                pass
            ds._index = ds._build_or_load_index()
            print(f"Rebuilt index, {len(ds._index)} records found")
    print(f"Loaded dataset with {len(ds)} records")

    n = min(5, len(ds))
    for i in range(n):
        try:
            x, y = ds[i]
        except Exception as exc:
            print(f"Record {i}: error while loading: {exc}")
            continue
        print(
            f"Record {i}: spectrum_len={x.numel()}, "
            f"binary_len={y['binary'].numel()}, "
            f"n_categorical={len(y['categorical'])}, "
            f"n_continuous={y['continuous'].numel()}"
        )


if __name__ == "__main__":
    main()
