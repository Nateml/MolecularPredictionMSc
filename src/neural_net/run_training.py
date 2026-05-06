from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

from loguru import logger
from torch.utils.data import Dataset

from config import config
from neural_net.model import MultitaskMLP
from neural_net.datasets import dataset_from_schema
from data.split import split_dataset
from neural_net.train import train

TRAIN_FRACTION = 0.75
VAL_FRACTION = 0.10
CALIBRATION_FRACTION = 0.05
TEST_FRACTION = 0.10

_SPLITS_DIR = Path(__file__).resolve().parents[2] / "data" / "splits"
_SPLIT_NAMES = ["train", "val", "calibration", "test"]


def _splits_exist(splits_dir: Path) -> bool:
    return all((splits_dir / f"{name}.pkl").exists() for name in _SPLIT_NAMES)


def _load_splits(splits_dir: Path) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    return tuple(
        pickle.loads((splits_dir / f"{name}.pkl").read_bytes())
        for name in _SPLIT_NAMES
    )


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
    model = MultitaskMLP(
        spectrum_len=config.model.spectrum_length,
        hidden_dims=config.model.hidden_dims,
        num_binary=dataset.maccs_length,
        categorical_sizes=dataset.categorical_sizes(),
        dropout=config.model.dropout,
    )

    if _splits_exist(_SPLITS_DIR):
        logger.info(f"Loading existing splits from {_SPLITS_DIR}...")
        train_ds, val_ds, calibration_ds, test_ds = _load_splits(_SPLITS_DIR)
    else:
        logger.info(f"No splits found at {_SPLITS_DIR}, creating new splits...")
        train_ds, val_ds, calibration_ds, test_ds = split_dataset(
            dataset,
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            calibration_fraction=CALIBRATION_FRACTION,
            test_fraction=TEST_FRACTION,
            output_dir=str(_SPLITS_DIR),
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
