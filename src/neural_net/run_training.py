from __future__ import annotations

import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

from loguru import logger
from torch.utils.data import Dataset

from config import config
from neural_net.model import MultitaskMLP, MultitaskMLPWithQRegression
from neural_net.datasets import dataset_from_schema, RawFeatureDataset, SpectrumDataset
from data.split import split_dataset
from neural_net.train import train

TRAIN_FRACTION = 0.75
VAL_FRACTION = 0.10
CALIBRATION_FRACTION = 0.05
TEST_FRACTION = 0.10
SEED = 42

_SPLITS_BASE = Path(__file__).resolve().parents[2] / "data" / "splits"
_SPLIT_NAMES = ["train", "val", "calibration", "test"]
_META_FILE = "meta.json"


def _splits_fingerprint(
    data_path: Path,
    schema_path: Path,
    fractions: Dict[str, float],
    seed: int,
) -> str:
    """Short hash identifying a unique (data, schema, fractions, seed) combination."""
    key = json.dumps({
        "data_path": str(data_path.resolve()),
        "schema_path": str(schema_path.resolve()),
        "fractions": fractions,
        "seed": seed,
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _splits_dir(
    data_path: Path,
    schema_path: Path,
    fractions: Dict[str, float],
    seed: int,
) -> Path:
    return _SPLITS_BASE / _splits_fingerprint(data_path, schema_path, fractions, seed)


def _save_splits_meta(
    splits_dir: Path,
    data_path: Path,
    schema_path: Path,
    dataset: SpectrumDataset,
    fractions: Dict[str, float],
    seed: int,
) -> None:
    meta = {
        "data_path": str(data_path.resolve()),
        "schema_path": str(schema_path.resolve()),
        "dataset_type": type(dataset).__name__,
        "n_records": len(dataset),
        "fractions": fractions,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (splits_dir / _META_FILE).write_text(json.dumps(meta, indent=2))


def _validate_splits_meta(
    splits_dir: Path,
    data_path: Path,
    schema_path: Path,
    fractions: Dict[str, float],
    seed: int,
) -> None:
    """Raise ValueError if saved meta doesn't match current parameters."""
    meta_path = splits_dir / _META_FILE
    if not meta_path.exists():
        raise ValueError(
            f"Splits directory {splits_dir} exists but has no {_META_FILE}. "
            "Delete the directory and re-run to regenerate splits."
        )
    meta = json.loads(meta_path.read_text())
    mismatches = []
    if meta["data_path"] != str(data_path.resolve()):
        mismatches.append(f"  data_path: saved={meta['data_path']!r}, current={str(data_path.resolve())!r}")
    if meta["schema_path"] != str(schema_path.resolve()):
        mismatches.append(f"  schema_path: saved={meta['schema_path']!r}, current={str(schema_path.resolve())!r}")
    if meta["fractions"] != fractions:
        mismatches.append(f"  fractions: saved={meta['fractions']}, current={fractions}")
    if meta["seed"] != seed:
        mismatches.append(f"  seed: saved={meta['seed']}, current={seed}")
    if mismatches:
        raise ValueError(
            f"Loaded splits at {splits_dir} do not match current config:\n"
            + "\n".join(mismatches)
        )


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
        logger.info("Dataset is not RawFeatureDataset, using standard MultitaskMLP model with binned continuous features.")
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
    splits_dir = _splits_dir(data_path, schema_path, fractions, SEED)

    if _splits_exist(splits_dir):
        logger.info(f"Loading existing splits from {splits_dir}...")
        _validate_splits_meta(splits_dir, data_path, schema_path, fractions, SEED)
        train_ds, val_ds, calibration_ds, test_ds = _load_splits(splits_dir)
    else:
        logger.info(f"No splits found at {splits_dir}, creating new splits...")
        splits_dir.mkdir(parents=True, exist_ok=True)
        train_ds, val_ds, calibration_ds, test_ds = split_dataset(
            dataset,
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            calibration_fraction=CALIBRATION_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=SEED,
            output_dir=str(splits_dir),
        )
        _save_splits_meta(splits_dir, data_path, schema_path, dataset, fractions, SEED)

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
