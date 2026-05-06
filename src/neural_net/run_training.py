from __future__ import annotations

import json
from typing import Dict

from loguru import logger
import torch

from src.config import config
from src.neural_net.model import MultitaskMLP
from src.neural_net.onehot_decoded_dataset import OneHotDecodedDataset
from src.data.split import split_dataset
from src.neural_net.train import train

TRAIN_FRACTION = 0.75
VAL_FRACTION = 0.10
CALIBRATION_FRACTION = 0.05
TEST_FRACTION = 0.10


def main():
    data_path = config.data.resolved_dataset_path()
    schema_path = config.data.resolved_schema_path()
    output_path = config.checkpointing.resolved_save_dir() / "multitask_mlp.pt"

    logger.info(f"Loading schema from {schema_path}...")
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    maccs_length = int(schema.get("maccs_length", 167))

    logger.info(f"Building dataset from {data_path}...")
    dataset = OneHotDecodedDataset(
        data_path=data_path,
        schema_path=schema_path,
        spectrum_len=config.model.spectrum_length,
    )

    logger.info("Deriving categorical_sizes from dataset and schema...")
    feature_schema = schema["feature_schema"]
    categorical_sizes: Dict[str, int] = {}
    for feat in dataset.feature_order:
        spec = feature_schema.get(feat)
        if spec is None:
            raise ValueError(f"Feature {feat} missing from schema.feature_schema")
        categorical_sizes[feat] = len(spec["onehot_names"])

    logger.info("Constructing model...")
    model = MultitaskMLP(
        spectrum_len=config.model.spectrum_length,
        hidden_dims=config.model.hidden_dims,
        num_binary=maccs_length,
        categorical_sizes=categorical_sizes,
        dropout=config.model.dropout,
    )

    logger.info("Splitting dataset into train/val/calibration/test...")
    train_ds, val_ds, calibration_ds, test_ds = split_dataset(
        dataset,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
        calibration_fraction=CALIBRATION_FRACTION,
        test_fraction=TEST_FRACTION,
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
