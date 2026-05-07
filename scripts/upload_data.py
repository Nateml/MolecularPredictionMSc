"""
Upload dataset files and a split to the HuggingFace dataset repo.

Usage:
    python scripts/upload_data.py --repo <org/repo-name>
    python scripts/upload_data.py --repo <org/repo-name> --split-hash f9d711f95583
    python scripts/upload_data.py --repo <org/repo-name> --dataset-only
    python scripts/upload_data.py --repo <org/repo-name> --splits-only --split-hash f9d711f95583
    python scripts/upload_data.py --repo <org/repo-name> \\
        --dataset-file mixed_raw_dataset.json \\
        --dataset-file feature_vector_mixed_schema.json

Requires HF_TOKEN env var or `huggingface-cli login` to have been run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

DEFAULT_DATASET_FILES = [
    "multitask_mlp_hardcoded.test.json",
    "feature_vector_mixed_schema.json",
]


def upload_dataset(api: HfApi, repo_id: str, dataset_files: list[str]) -> None:
    for filename in dataset_files:
        local = DATA_DIR / filename
        if not local.exists():
            print(f"  Skipping {filename} (not found locally)", file=sys.stderr)
            continue
        print(f"  Uploading {filename} ({local.stat().st_size / 1e6:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(local),
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"  Done: {filename}")


def upload_split(api: HfApi, repo_id: str, split_hash: str) -> None:
    split_dir = SPLITS_DIR / split_hash
    if not split_dir.exists():
        print(f"Split {split_hash} not found at {split_dir}", file=sys.stderr)
        sys.exit(1)

    files = list(split_dir.iterdir())
    print(f"  Uploading {len(files)} files for split {split_hash}...")
    for f in sorted(files):
        repo_path = f"splits/{split_hash}/{f.name}"
        print(f"  Uploading {repo_path} ({f.stat().st_size / 1e3:.1f} KB)...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
        )
    print(f"  Done: split {split_hash}")


def list_local_splits() -> list[str]:
    if not SPLITS_DIR.exists():
        return []
    return [d.name for d in sorted(SPLITS_DIR.iterdir()) if d.is_dir()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="HuggingFace repo ID, e.g. myorg/mol-dataset")
    p.add_argument("--split-hash", help="Split hash to upload (e.g. f9d711f95583). Required unless --dataset-only.")
    p.add_argument(
        "--dataset-file", dest="dataset_files", metavar="FILE", action="append",
        help=f"Dataset filename to upload from data/ (repeatable). Defaults: {DEFAULT_DATASET_FILES}",
    )
    p.add_argument("--dataset-only", action="store_true", help="Upload dataset files only, skip splits")
    p.add_argument("--splits-only", action="store_true", help="Upload split only, skip dataset files")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HuggingFace API token (default: $HF_TOKEN)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dataset_only and not args.split_hash:
        local_splits = list_local_splits()
        if not local_splits:
            print("No splits found locally and --split-hash not specified.", file=sys.stderr)
            sys.exit(1)
        if len(local_splits) == 1:
            args.split_hash = local_splits[0]
            print(f"Auto-selected split: {args.split_hash}")
        else:
            print("Multiple local splits found. Specify one with --split-hash:", file=sys.stderr)
            for s in local_splits:
                print(f"  {s}", file=sys.stderr)
            sys.exit(1)

    dataset_files = args.dataset_files or DEFAULT_DATASET_FILES

    api = HfApi(token=args.token)

    # Ensure repo exists (no-op if already exists)
    create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True, token=args.token)
    print(f"Repo: https://huggingface.co/datasets/{args.repo}")

    if not args.splits_only:
        print("Uploading dataset files...")
        upload_dataset(api, args.repo, dataset_files)

    if not args.dataset_only:
        print(f"Uploading split {args.split_hash}...")
        upload_split(api, args.repo, args.split_hash)

    print("Upload complete.")


if __name__ == "__main__":
    main()
