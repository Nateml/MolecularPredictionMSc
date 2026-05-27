from __future__ import annotations

import hashlib
import json
import pickle
import random
import torch
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Tuple
from loguru import logger
from torch.utils.data import random_split

from torch.utils.data import Subset
from src.data.datasets import SpectrumDataset

TRAIN_FRACTION: Final[float] = 0.75
VAL_FRACTION: Final[float] = 0.10
CALIBRATION_FRACTION: Final[float] = 0.05
TEST_FRACTION: Final[float] = 0.10
SEED: Final[int] = 42

_SPLITS_BASE = Path(__file__).resolve().parents[2] / "data" / "splits"
_SPLIT_NAMES = ["train", "val", "calibration", "test"]
_META_FILE = "meta.json"


# ---------------------------------------------------------------------------
# Public split-directory helpers
# ---------------------------------------------------------------------------

def splits_fingerprint(
    data_path: Path,
    schema_path: Path,
    fractions: dict[str, float],
    seed: int,
) -> str:
    key = json.dumps({
        "data_path": str(data_path.resolve()),
        "schema_path": str(schema_path.resolve()),
        "fractions": fractions,
        "seed": seed,
    }, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def splits_dir(
    data_path: Path,
    schema_path: Path,
    fractions: dict[str, float],
    seed: int,
) -> Path:
    return _SPLITS_BASE / splits_fingerprint(data_path, schema_path, fractions, seed)


def splits_exist(directory: Path) -> bool:
    return (directory / "dataset.pkl").exists() and all(
        (directory / f"{name}.pkl").exists() for name in _SPLIT_NAMES
    )


def load_splits(directory: Path) -> Tuple[SpectrumDataset, SpectrumDataset, SpectrumDataset, SpectrumDataset]:
    dataset = pickle.loads((directory / "dataset.pkl").read_bytes())
    return tuple(
        Subset(dataset, pickle.loads((directory / f"{name}.pkl").read_bytes()))
        for name in _SPLIT_NAMES
    )


def validate_splits_meta(
    directory: Path,
    data_path: Path,
    schema_path: Path,
    fractions: dict[str, float],
    seed: int,
) -> None:
    """Raise ValueError if saved meta.json doesn't match current parameters."""
    meta_path = directory / _META_FILE
    if not meta_path.exists():
        raise ValueError(
            f"Splits directory {directory} exists but has no {_META_FILE}. "
            "Delete the directory and re-run to regenerate splits."
        )
    meta = json.loads(meta_path.read_text())
    mismatches = []

    if meta["data_path"] != str(data_path):
        mismatches.append(f"  data_path: saved={meta['data_path']!r}, current={str(data_path)!r}")
    if meta["schema_path"] != str(schema_path):
        mismatches.append(f"  schema_path: saved={meta['schema_path']!r}, current={str(schema_path)!r}")
    if meta["fractions"] != fractions:
        mismatches.append(f"  fractions: saved={meta['fractions']}, current={fractions}")
    if meta["seed"] != seed:
        mismatches.append(f"  seed: saved={meta['seed']}, current={seed}")
    if mismatches:
        raise ValueError(
            f"Loaded splits at {directory} do not match current config:\n"
            + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def split_dataset_old(
    dataset: SpectrumDataset,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = 42,
) -> tuple[SpectrumDataset, SpectrumDataset, SpectrumDataset, SpectrumDataset]:
    """
    Split the dataset into train/val/calibration/test subsets, not taking into account SMILES duplicates or label consistency. This is a fallback method and should not be used if the dataset has SMILES records.
    """

    train_ds, val_ds, calibration_ds, test_ds = random_split(
        dataset,
        [train_fraction, val_fraction, calibration_fraction, test_fraction],
        generator=torch.Generator().manual_seed(seed),
    )

    train_idx = train_ds.indices
    val_idx = val_ds.indices
    calibration_idx = calibration_ds.indices
    test_idx = test_ds.indices

    split_indices = {
        "train": train_idx,
        "val": val_idx,
        "calibration": calibration_idx,
        "test": test_idx,
    }

    fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "calibration": calibration_fraction,
        "test": test_fraction,
    }
    split_names = fractions.keys()
    out_dir = splits_dir(dataset.data_path, dataset.schema_path, fractions, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the dataset once (cache excluded via __getstate__) + per-split index lists.
    # This avoids serialising the full dataset object once per split.
    logger.info(f"Saving splits to {out_dir}...")
    with open(out_dir / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    for name in split_names:
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(split_indices[name], f)
            print(f"Saved split '{name}' with {len(split_indices[name])} records.")
    _save_splits_meta(out_dir, dataset.data_path, dataset.schema_path, dataset, fractions, seed)

    return tuple(
        Subset(dataset, split_indices[split_name])
        for split_name in split_names
    )


def split_dataset(
    dataset: SpectrumDataset,
    train_fraction: float = 0.75,
    val_fraction: float = 0.10,
    calibration_fraction: float = 0.05,
    test_fraction: float = 0.10,
    seed: int = 42,
    check_label_consistency: bool = True,
) -> tuple[SpectrumDataset, SpectrumDataset, SpectrumDataset, SpectrumDataset]:

    fractions = [train_fraction, val_fraction, calibration_fraction, test_fraction]
    fraction_total = sum(fractions)
    if abs(fraction_total - 1.0) > 1e-6:
        raise ValueError(
            "Split fractions must sum to 1.0, got "
            f"{fraction_total:.6f}"
        )

    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError(
            "split_dataset requires a dataset with a `records` attribute containing SMILES fields."
        )
    #print length of records
    print(f"Dataset has {len(records)} records to split.")

    grouped_records: dict[str, list[int]] = defaultdict(list)
    for index, record in tqdm(enumerate(records), desc="Grouping records by SMILES"):
        grouped_records[_record_smiles(record, index)].append(index)

    grouped_indices: list[list[int]] = []
    strata_keys: list[tuple[int, tuple[int, ...]]] = []
    for indices in tqdm(grouped_records.values(), desc="Processing groups for stratification"):
        grouped_indices.append(indices)
        _, reference_targets = dataset[indices[0]]
        if check_label_consistency:
            for index in indices[1:]:
                _, targets = dataset[index]
                if not _targets_match(reference_targets, targets):
                    raise ValueError(
                        "Identical SMILES records must have identical labels, but a duplicate group contains conflicting targets."
                    )
        strata_keys.append(_stratification_key(reference_targets))

    split_names = ["train", "val", "calibration", "test"]
    split_indices = _allocate_grouped_strata(
        grouped_indices=grouped_indices,
        strata_keys=strata_keys,
        split_names=split_names,
        split_fractions=fractions,
        seed=seed,
    )

    fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "calibration": calibration_fraction,
        "test": test_fraction,
    }
    out_dir = splits_dir(dataset.data_path, dataset.schema_path, fractions, seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the dataset once (cache excluded via __getstate__) + per-split index lists.
    # This avoids serialising the full dataset object once per split.
    logger.info(f"Saving splits to {out_dir}...")
    with open(out_dir / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    for name in split_names:
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(split_indices[name], f)
            print(f"Saved split '{name}' with {len(split_indices[name])} records.")
    _save_splits_meta(out_dir, dataset.data_path, dataset.schema_path, dataset, fractions, seed)

    return tuple(
        Subset(dataset, split_indices[split_name])
        for split_name in split_names
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_splits_meta(
    directory: Path,
    data_path: Path,
    schema_path: Path,
    dataset: SpectrumDataset,
    fractions: dict[str, float],
    seed: int,
) -> None:
    meta = {
        "data_path": str(data_path),
        "schema_path": str(schema_path),
        "dataset_type": type(dataset).__name__,
        "n_records": len(dataset),
        "fractions": fractions,
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (directory / _META_FILE).write_text(json.dumps(meta, indent=2))


def _record_smiles(record: object, index: int) -> str:
    if isinstance(record, dict):
        smiles = record.get("smiles")
        if smiles is None:
            compound = record.get("compound")
            if isinstance(compound, dict):
                smiles = compound.get("smiles")
        if smiles is not None:
            smiles_text = str(smiles).strip()
            if smiles_text:
                return smiles_text
    return f"__record_{index}"

def _stratification_key(targets: dict[str, torch.Tensor]) -> tuple[int, tuple[int, ...]]:
    binary_active_count = int(torch.as_tensor(targets["binary"]).sum().item())
    categorical_key = tuple(int(target.item()) for target in targets["categorical"])
    return binary_active_count, categorical_key


def _targets_match(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if not torch.equal(left["binary"], right["binary"]):
        return False
    if len(left["categorical"]) != len(right["categorical"]):
        return False
    return all(torch.equal(left_item, right_item) for left_item, right_item in zip(left["categorical"], right["categorical"]))

def _allocate_grouped_strata(
    grouped_indices: list[list[int]],
    strata_keys: list[tuple[int, tuple[int, ...]]],
    split_names: list[str],
    split_fractions: list[float],
    seed: int,
) -> dict[str, list[int]]:
    total_records = sum(len(indices) for indices in grouped_indices)
    # Use integer desired counts (rounded) to avoid tiny fractional targets
    desired_counts = {
        split_name: int(round(total_records * fraction)) if fraction > 0 else 0
        for split_name, fraction in zip(split_names, split_fractions)
    }

    stratum_totals: dict[tuple[int, tuple[int, ...]], int] = defaultdict(int)
    for indices, strata_key in zip(grouped_indices, strata_keys):
        stratum_totals[strata_key] += len(indices)

    desired_stratum_totals: dict[str, dict[tuple[int, tuple[int, ...]], float]] = {
        split_name: {
            strata_key: total * fraction
            for strata_key, total in stratum_totals.items()
        }
        for split_name, fraction in zip(split_names, split_fractions)
    }

    split_records: dict[str, list[int]] = {split_name: [] for split_name in split_names}
    split_totals: dict[str, int] = {split_name: 0 for split_name in split_names}
    split_stratum_totals: dict[str, dict[tuple[int, tuple[int, ...]], int]] = {
        split_name: defaultdict(int) for split_name in split_names
    }

    rng = random.Random(seed)
    shuffled_groups = list(zip(grouped_indices, strata_keys))
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=lambda item: len(item[0]), reverse=True)

    for indices, strata_key in shuffled_groups:
        group_size = len(indices)
        best_split_name = split_names[0]
        best_score = None

        for split_name in split_names:
            prospective_total = split_totals[split_name] + group_size
            # Compare against integer desired counts when possible
            target_total = desired_counts.get(split_name, total_records * split_fractions[split_names.index(split_name)])
            total_score = abs(prospective_total - target_total)

            prospective_stratum_total = split_stratum_totals[split_name][strata_key] + group_size
            stratum_score = abs(
                prospective_stratum_total - desired_stratum_totals[split_name][strata_key]
            )

            score = total_score + stratum_score
            if best_score is None or score < best_score:
                best_score = score
                best_split_name = split_name

        split_records[best_split_name].extend(indices)
        split_totals[best_split_name] += group_size
    # If any split with a positive desired count ended up empty, fall back to
    # a deterministic grouped round-robin to guarantee at least some records
    # are assigned to each requested split.
    counts = {name: len(split_records[name]) for name in split_names}
    need_fix = any(desired_counts.get(name, 0) > 0 and counts.get(name, 0) == 0 for name in split_names)
    if need_fix:
        # Rebuild assignment deterministically to respect grouped units
        split_records = {name: [] for name in split_names}
        split_totals = {name: 0 for name in split_names}
        # Desired counts sum may not equal total_records due to rounding; compute target per split
        targets = dict(desired_counts)
        remaining = total_records - sum(targets.values())
        if remaining > 0:
            # add remaining to the largest-fraction split
            max_idx = max(range(len(split_fractions)), key=lambda i: split_fractions[i])
            targets[split_names[max_idx]] += remaining

        for grp in grouped_indices:
            # pick first split that still needs records, else the split with largest fraction
            placed = False
            for name in split_names:
                if split_totals[name] < targets.get(name, 0):
                    split_records[name].extend(grp)
                    split_totals[name] += len(grp)
                    placed = True
                    break
            if not placed:
                # all targets satisfied; append to the split with largest fraction
                split_records[split_names[max_idx]].extend(grp)
                split_totals[split_names[max_idx]] += len(grp)
        split_stratum_totals[best_split_name][strata_key] += group_size

    return split_records


# ---------------------------------------------------------------------------
# Entry point for splitting the dataset before training
# ---------------------------------------------------------------------------

def main():
    from config import config
    from src.data.datasets import dataset_from_schema

    data_path = config.data.resolved_dataset_path()
    schema_path = config.data.resolved_schema_path()
    logger.info(f"Building dataset from {data_path}...")
    dataset = dataset_from_schema(
        data_path=data_path,
        schema_path=schema_path,
        spectrum_len=config.model.spectrum_length,
    )
    logger.info(f"Dataset type: {type(dataset).__name__}, {len(dataset)} records")

    fractions = {
        "train": TRAIN_FRACTION,
        "val": VAL_FRACTION,
        "calibration": CALIBRATION_FRACTION,
        "test": TEST_FRACTION,
    }
    split_dir = splits_dir(data_path, schema_path, fractions, SEED)
    if splits_exist(split_dir):
        # Ask user to confirm overwrite of existing splits
        logger.warning(f"Splits already exist at {split_dir}.")
        response = input("Do you want to overwrite them? (y/N) ").strip().lower()
        if response != "y":
            logger.info("Aborting without overwriting splits.")
            return
        else:
            logger.info("Overwriting existing splits...")

    logger.info("Splitting dataset...")
    _, _, _, _ = split_dataset_old(
        dataset,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
        calibration_fraction=CALIBRATION_FRACTION,
        test_fraction=TEST_FRACTION,
        seed=SEED,
    )

    logger.info("Splitting complete.")
    logger.info("Splits saved to: " + str(split_dir))
    logger.info("Load splits in training with `load_splits(split_dir)`.")

if __name__ == "__main__":
    main()
