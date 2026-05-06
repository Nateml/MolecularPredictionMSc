from __future__ import annotations

import json

from pathlib import Path
from loguru import logger

from config import config
from model.model import MultitaskMLP, MultitaskMLPWithQRegression
from data.datasets import dataset_from_schema, RawFeatureDataset, SpectrumDataset
from data.split import (
    split_dataset,
    splits_dir,
    splits_exist,
    load_splits,
    validate_splits_meta,
    TRAIN_FRACTION,
    VAL_FRACTION,
    CALIBRATION_FRACTION,
    TEST_FRACTION,
    SEED
)
from model.train import train

def main():
    data_path = config.data.resolved_dataset_path()
    schema_path = config.data.resolved_schema_path()
    output_path = config.checkpointing.resolved_save_dir() / "multitask_mlp.pt"

    logger.info(f"Building dataset from {data_path}...")
    dataset = dataset_from_schema(
        data_path=data_path,
        schema_path=schema_path,
        spectrum_len=config.model.spectrum_length,
    )
    logger.info(f"Dataset type: {type(dataset).__name__}, {len(dataset)} records")

    logger.info("Constructing model...")
    if isinstance(dataset, RawFeatureDataset):
        logger.info("Dataset is RawFeatureDataset, using QRegression model.")
        model = MultitaskMLPWithQRegression(
            spectrum_len=config.model.spectrum_length,
            hidden_dims=config.model.hidden_dims,
            num_binary=dataset.maccs_length,
            categorical_sizes=dataset.categorical_sizes(),
            num_quantiles=config.model.num_quantiles,
            regression_hidden_dim=config.model.regression_hidden_dim,
            dropout=config.model.dropout,
        )
    else:
        logger.info("Dataset is not RawFeatureDataset, using standard MultitaskMLP.")
        model = MultitaskMLP(
            spectrum_len=config.model.spectrum_length,
            hidden_dims=config.model.hidden_dims,
            num_binary=dataset.maccs_length,
            categorical_sizes=dataset.categorical_sizes(),
            dropout=config.model.dropout,
        )

    fractions = {
        "train": TRAIN_FRACTION,
        "val": VAL_FRACTION,
        "calibration": CALIBRATION_FRACTION,
        "test": TEST_FRACTION,
    }
    split_dir = splits_dir(data_path, schema_path, fractions, SEED)

    if splits_exist(split_dir):
        logger.info(f"Loading existing splits from {split_dir}...")
        validate_splits_meta(split_dir, data_path, schema_path, fractions, SEED)
        train_ds, val_ds, calibration_ds, test_ds = load_splits(split_dir)
    else:
        logger.info(f"No splits found at {split_dir}, creating new splits...")
        train_ds, val_ds, calibration_ds, test_ds = split_dataset(
            dataset,
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            calibration_fraction=CALIBRATION_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=SEED,
            check_label_consistency=True,
        )

    logger.info(
        f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, "
        f"Calibration samples: {len(calibration_ds)}, Test samples: {len(test_ds)}"
    )

    # hash of this specific data split
    split_hash = split_dir.name

    # Dataset is either of type RawFeatureDataset or PackedDataset, both of which
    # inherit from SpectrumDataset which implements from_subset.
    _class = type(dataset)
    train_ds = _class.from_subset(train_ds)
    val_ds = _class.from_subset(val_ds)
    calibration_ds = _class.from_subset(calibration_ds)
    test_ds = _class.from_subset(test_ds)

    norm_stats = None
    if isinstance(model, MultitaskMLPWithQRegression):
        logger.info("Normalising regression targets...")
        mean, std = train_ds.normalise_continuous()
        _, _ = val_ds.normalise_continuous(mean, std)
        _, _ = calibration_ds.normalise_continuous(mean, std)
        _, _ = test_ds.normalise_continuous(mean, std)
        # Save normalization stats for later use in evaluation
        norm_stats = {
            "mean": mean,
            "std": std
        }
        norm_stats_path = split_dir / "normalization.json"
        with open(norm_stats_path, "w", encoding="utf-8") as f:
            json.dump(norm_stats, f)
        logger.info(f"Saved normalization stats to {norm_stats_path}")


    # Modify output path to include split hash as final directory
    output_path = output_path.parent / split_hash / output_path.name

    logger.info(f"Training with output path: {output_path}...")
    train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        n_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        lr=config.training.learning_rate,
        output_path=output_path,
        continuous_stats=norm_stats if isinstance(model, MultitaskMLPWithQRegression) else None,
    )


if __name__ == "__main__":
    main()
