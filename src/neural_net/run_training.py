from __future__ import annotations

from pathlib import Path

from loguru import logger

from config import config
from neural_net.model import MultitaskMLP, MultitaskMLPWithQRegression
from neural_net.datasets import dataset_from_schema, RawFeatureDataset
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
)
from neural_net.train import train

SEED = 42


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
        )

    logger.info(
        f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, "
        f"Calibration samples: {len(calibration_ds)}, Test samples: {len(test_ds)}"
    )
    train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        n_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        lr=config.training.learning_rate,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
