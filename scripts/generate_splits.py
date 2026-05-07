"""
Generate and cache dataset splits, then optionally upload them to HuggingFace.

Uses the same fractions and seed as run_training.py so the resulting split hash
matches what training expects.

Usage:
    python scripts/generate_splits.py
    python scripts/generate_splits.py --upload --repo myorg/mol-dataset
    python scripts/generate_splits.py --data-path data/mixed_raw_dataset.json \\
        --schema-path data/feature_vector_mixed_schema.json \\
        --upload --repo myorg/mol-dataset

Requires HF_TOKEN env var or `huggingface-cli login` if --upload is used.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC))

from config import config
from data.datasets import dataset_from_schema
from data.split import (
    CALIBRATION_FRACTION,
    SEED,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VAL_FRACTION,
    split_dataset,
    splits_dir,
    splits_exist,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--data-path", default=None,
        help="Path to dataset JSON (default: config.yaml data.dataset_path)",
    )
    p.add_argument(
        "--schema-path", default=None,
        help="Path to schema JSON (default: config.yaml data.schema_path)",
    )
    p.add_argument("--upload", action="store_true", help="Upload generated split to HuggingFace after creation")
    p.add_argument("--repo", help="HuggingFace repo ID (required with --upload)")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HuggingFace API token (default: $HF_TOKEN)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.upload and not args.repo:
        print("Error: --repo is required when using --upload.", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data_path) if args.data_path else config.data.resolved_dataset_path()
    schema_path = Path(args.schema_path) if args.schema_path else config.data.resolved_schema_path()

    fractions = {
        "train": TRAIN_FRACTION,
        "val": VAL_FRACTION,
        "calibration": CALIBRATION_FRACTION,
        "test": TEST_FRACTION,
    }

    split_dir = splits_dir(data_path, schema_path, fractions, SEED)
    split_hash = split_dir.name

    if splits_exist(split_dir):
        print(f"Splits already exist: {split_dir}")
        print(f"Split hash: {split_hash}")
    else:
        print(f"Building dataset from {data_path}...")
        dataset = dataset_from_schema(
            data_path=data_path,
            schema_path=schema_path,
            spectrum_len=1000,
        )
        print(f"Dataset loaded: {len(dataset)} records. Generating splits...")
        split_dataset(
            dataset,
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            calibration_fraction=CALIBRATION_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=SEED,
        )
        print(f"Splits saved to {split_dir}")
        print(f"Split hash: {split_hash}")

    if args.upload:
        print(f"Uploading split {split_hash} to {args.repo}...")
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "upload_data.py"),
            "--repo", args.repo,
            "--split-hash", split_hash,
            "--splits-only",
        ]
        if args.token:
            cmd += ["--token", args.token]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
