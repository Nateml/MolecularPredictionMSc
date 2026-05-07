"""
Fetch dataset files and a split from the HuggingFace dataset repo.

Usage:
    python scripts/fetch_data.py --repo <org/repo-name> --split-hash f9d711f95583
    python scripts/fetch_data.py --repo <org/repo-name> --list-splits
    python scripts/fetch_data.py --repo <org/repo-name> --dataset-only
    python scripts/fetch_data.py --repo <org/repo-name> --split-hash f9d711f95583 --splits-only
    python scripts/fetch_data.py --repo <org/repo-name> \\
        --dataset-file mixed_raw_dataset.json \\
        --dataset-file feature_vector_mixed_schema.json

Files are written to data/ and data/splits/<hash>/ under the project root.
Already-present files are skipped unless --force is given.

Requires HF_TOKEN env var or `huggingface-cli login` to have been run.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, list_repo_tree

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"

DEFAULT_DATASET_FILES = [
    "multitask_mlp_hardcoded.test.json",
    "feature_vector_mixed_schema.json",
]


def fetch_dataset(api: HfApi, repo_id: str, token: str | None, force: bool, dataset_files: list[str]) -> None:
    for filename in dataset_files:
        dest = DATA_DIR / filename
        if dest.exists() and not force:
            print(f"  Skipping {filename} (already present; use --force to overwrite)")
            continue
        print(f"  Downloading {filename}...")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=token,
            local_dir=str(DATA_DIR),
        )
        print(f"  Saved: {path}")


def fetch_split(api: HfApi, repo_id: str, split_hash: str, token: str | None, force: bool) -> None:
    split_dir = SPLITS_DIR / split_hash
    split_dir.mkdir(parents=True, exist_ok=True)

    # List files in splits/<hash>/ on the remote
    try:
        remote_files = [
            item.path
            for item in list_repo_tree(repo_id, repo_type="dataset", token=token)
            if item.path.startswith(f"splits/{split_hash}/")
        ]
    except Exception as e:
        print(f"Failed to list remote split files: {e}", file=sys.stderr)
        sys.exit(1)

    if not remote_files:
        print(f"No files found for split {split_hash} in repo {repo_id}.", file=sys.stderr)
        sys.exit(1)

    print(f"  Downloading {len(remote_files)} files for split {split_hash}...")
    for repo_path in sorted(remote_files):
        filename = Path(repo_path).name
        dest = split_dir / filename
        if dest.exists() and not force:
            print(f"  Skipping {filename} (already present)")
            continue
        print(f"  Downloading {repo_path}...")
        hf_hub_download(
            repo_id=repo_id,
            filename=repo_path,
            repo_type="dataset",
            token=token,
            local_dir=str(DATA_DIR),
        )
    print(f"  Done: split at {split_dir}")


def list_remote_splits(repo_id: str, token: str | None) -> list[str]:
    seen: set[str] = set()
    for item in list_repo_tree(repo_id, repo_type="dataset", token=token):
        parts = Path(item.path).parts
        if len(parts) >= 2 and parts[0] == "splits":
            seen.add(parts[1])
    return sorted(seen)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="HuggingFace repo ID, e.g. myorg/mol-dataset")
    p.add_argument("--split-hash", help="Split hash to download (e.g. f9d711f95583)")
    p.add_argument(
        "--dataset-file", dest="dataset_files", metavar="FILE", action="append",
        help=f"Dataset filename to fetch into data/ (repeatable). Defaults: {DEFAULT_DATASET_FILES}",
    )
    p.add_argument("--list-splits", action="store_true", help="List available splits in the repo and exit")
    p.add_argument("--dataset-only", action="store_true", help="Download dataset files only, skip splits")
    p.add_argument("--splits-only", action="store_true", help="Download split only, skip dataset files")
    p.add_argument("--force", action="store_true", help="Overwrite existing local files")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HuggingFace API token (default: $HF_TOKEN)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api = HfApi(token=args.token)

    if args.list_splits:
        splits = list_remote_splits(args.repo, args.token)
        if splits:
            print(f"Available splits in {args.repo}:")
            for s in splits:
                print(f"  {s}")
        else:
            print(f"No splits found in {args.repo}.")
        return

    if not args.dataset_only and not args.split_hash:
        # Auto-select if only one split exists remotely
        splits = list_remote_splits(args.repo, args.token)
        if not splits:
            print("No splits found in repo. Use --dataset-only or upload a split first.", file=sys.stderr)
            sys.exit(1)
        if len(splits) == 1:
            args.split_hash = splits[0]
            print(f"Auto-selected split: {args.split_hash}")
        else:
            print("Multiple splits available. Specify one with --split-hash:", file=sys.stderr)
            for s in splits:
                print(f"  {s}", file=sys.stderr)
            sys.exit(1)

    dataset_files = args.dataset_files or DEFAULT_DATASET_FILES
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.splits_only:
        print("Downloading dataset files...")
        fetch_dataset(api, args.repo, args.token, args.force, dataset_files)

    if not args.dataset_only:
        print(f"Downloading split {args.split_hash}...")
        fetch_split(api, args.repo, args.split_hash, args.token, args.force)

    print("Fetch complete.")


if __name__ == "__main__":
    main()
