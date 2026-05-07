#!/usr/bin/env python
"""
Compute conformal prediction thresholds for a pretrained checkpoint.

Loads the model, runs the calibration split through it, computes
nonconformity score thresholds, and saves conformal_thresholds.pt
in the same directory as the checkpoint.

Usage:
    python calibrate.py
    python calibrate.py --checkpoint models/final/multitask_mlp_mixed_best.pt
    python calibrate.py --checkpoint models/final/multitask_mlp_mixed_best.pt --alpha 0.05
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import config
from data.datasets import RawFeatureDataset
from data.split import (
    splits_dir,
    splits_exist,
    load_splits,
    validate_splits_meta,
    TRAIN_FRACTION,
    VAL_FRACTION,
    CALIBRATION_FRACTION,
    TEST_FRACTION,
    SEED,
)
from infer import load_model
from model.conformal import calibrate_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute conformal thresholds for a pretrained checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to .pt checkpoint. Defaults to models/multitask_mlp.best.pt",
    )
    p.add_argument(
        "--alpha", type=float, default=config.conformal.alpha,
        help=f"Miscoverage level; target coverage = 1 - alpha (default: {config.conformal.alpha})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        config.checkpointing.resolved_save_dir() / "multitask_mlp.best.pt"
    )
    if not checkpoint_path.exists():
        checkpoint_path = config.checkpointing.resolved_save_dir() / "multitask_mlp.latest.pt"
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model from {checkpoint_path}...")
    model, dataset, device = load_model(checkpoint_path)

    data_path = config.data.resolved_dataset_path()
    schema_path = config.data.resolved_schema_path()
    fractions = {
        "train": TRAIN_FRACTION,
        "val": VAL_FRACTION,
        "calibration": CALIBRATION_FRACTION,
        "test": TEST_FRACTION,
    }

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    split_hash = ckpt.get("split_hash", None)
    if split_hash:
        split_dir = data_path.parent / "splits" / split_hash
    else:
        print("Warning: No split hash found in checkpoint. Attempting to find splits by data/schema/fraction config, but this may fail if the splits were generated with different config or code version.", file=sys.stderr)
        split_dir = splits_dir(data_path, schema_path, fractions, SEED)

    if not splits_exist(split_dir):
        print(
            f"No splits found at {split_dir}. "
            "Run training first (python -m src.model.run_training) to generate splits.",
            file=sys.stderr,
        )
        sys.exit(1)

    validate_splits_meta(split_dir, data_path, schema_path, fractions, SEED)
    _, _, calibration_subset, _ = load_splits(split_dir)

    _class = type(dataset)
    calibration_ds = _class.from_subset(calibration_subset)

    # Re-apply normalization stats from the loaded dataset (cleared by from_subset)
    if isinstance(calibration_ds, RawFeatureDataset) and dataset._cont_mean is not None:
        calibration_ds.normalise_continuous(
            dataset._cont_mean.tolist(),
            dataset._cont_std.tolist(),
        )

    print(f"Calibrating on {len(calibration_ds)} samples with alpha={args.alpha}...")
    calib_loader = DataLoader(calibration_ds, batch_size=config.training.batch_size, shuffle=False)
    thresholds = calibrate_model(model, calib_loader, alpha=args.alpha)

    out_path = checkpoint_path.parent / "conformal_thresholds.pt"
    torch.save(thresholds, out_path)
    print(f"Conformal thresholds saved to {out_path}")


if __name__ == "__main__":
    main()
