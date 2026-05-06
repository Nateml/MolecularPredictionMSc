from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import torch
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Tuple

from torch.utils.data import Subset
from neural_net.datasets import SpectrumDataset

TRAIN_FRACTION: Final[float] = 0.75
VAL_FRACTION: Final[float] = 0.10
CALIBRATION_FRACTION: Final[float] = 0.05
TEST_FRACTION: Final[float] = 0.10

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
    return all((directory / f"{name}.pkl").exists() for name in _SPLIT_NAMES)


def load_splits(directory: Path) -> Tuple[SpectrumDataset, SpectrumDataset, SpectrumDataset, SpectrumDataset]:
    return tuple(
        pickle.loads((directory / f"{name}.pkl").read_bytes())
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
            f"Loaded splits at {directory} do not match current config:\n"
            + "\n".join(mismatches)
        )


# ---------------------------------------------------------------------------
# Main split function
# ---------------------------------------------------------------------------

def split_dataset(
    dataset: SpectrumDataset,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = 42,
    check_label_consistency: bool = True,
) -> tuple[SpectrumDataset, SpectrumDataset, SpectrumDataset, SpectrumDataset]:
    """
    Split dataset into train/val/calibration/test subsets, grouped by SMILES to
    prevent data leakage. Saves splits to a hash-keyed directory under data/splits/
    so different datasets never share cached splits.
    """
    fractions_list = [train_fraction, val_fraction, calibration_fraction, test_fraction]
    fraction_total = sum(fractions_list)
    if abs(fraction_total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {fraction_total:.6f}")

    records = getattr(dataset, "records", None)
    if records is None:
        raise ValueError(
            "split_dataset requires a dataset with a `records` attribute containing SMILES fields."
        )

    grouped_records: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped_records[_record_smiles(record, index)].append(index)

    grouped_indices: list[list[int]] = []
    strata_keys: list[tuple[int, tuple[int, ...]]] = []
    for indices in grouped_records.values():
        grouped_indices.append(indices)
        _, reference_targets = dataset[indices[0]]
        if check_label_consistency:
            for index in indices[1:]:
                _, targets = dataset[index]
                if not _targets_match(reference_targets, targets):
                    raise ValueError(
                        "Identical SMILES records must have identical labels, "
                        "but a duplicate group contains conflicting targets."
                    )
        strata_keys.append(_stratification_key(reference_targets))

    split_names = ["train", "val", "calibration", "test"]
    split_indices = _allocate_grouped_strata(
        grouped_indices=grouped_indices,
        strata_keys=strata_keys,
        split_names=split_names,
        split_fractions=fractions_list,
        seed=seed,
    )

    subsets = {
        name: Subset(dataset, split_indices[name])
        for name in split_names
    }

    fractions = {
        "train": train_fraction,
        "val": val_fraction,
        "calibration": calibration_fraction,
        "test": test_fraction,
    }
    out_dir = splits_dir(dataset.data_path, dataset.schema_path, fractions, seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, subset in subsets.items():
        with open(out_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(subset, f)
    _save_splits_meta(out_dir, dataset.data_path, dataset.schema_path, dataset, fractions, seed)

    return tuple(subsets[name] for name in split_names)


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
        "data_path": str(data_path.resolve()),
        "schema_path": str(schema_path.resolve()),
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
    return all(torch.equal(l, r) for l, r in zip(left["categorical"], right["categorical"]))


def _allocate_grouped_strata(
    grouped_indices: list[list[int]],
    strata_keys: list[tuple[int, tuple[int, ...]]],
    split_names: list[str],
    split_fractions: list[float],
    seed: int,
) -> dict[str, list[int]]:
    total_records = sum(len(i) for i in grouped_indices)
    desired_totals = {
        name: total_records * frac
        for name, frac in zip(split_names, split_fractions)
    }

    stratum_totals: dict[tuple[int, tuple[int, ...]], int] = defaultdict(int)
    for indices, key in zip(grouped_indices, strata_keys):
        stratum_totals[key] += len(indices)

    desired_stratum_totals = {
        name: {key: total * frac for key, total in stratum_totals.items()}
        for name, frac in zip(split_names, split_fractions)
    }

    split_records: dict[str, list[int]] = {name: [] for name in split_names}
    split_totals: dict[str, int] = {name: 0 for name in split_names}
    split_stratum_totals: dict[str, dict] = {name: defaultdict(int) for name in split_names}

    rng = random.Random(seed)
    shuffled_groups = list(zip(grouped_indices, strata_keys))
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=lambda item: len(item[0]), reverse=True)

    for indices, key in shuffled_groups:
        group_size = len(indices)
        best_name = split_names[0]
        best_score = None

        for name in split_names:
            total_score = abs(split_totals[name] + group_size - desired_totals[name])
            stratum_score = abs(split_stratum_totals[name][key] + group_size - desired_stratum_totals[name][key])
            score = total_score + stratum_score
            if best_score is None or score < best_score:
                best_score = score
                best_name = name

        split_records[best_name].extend(indices)
        split_totals[best_name] += group_size
        split_stratum_totals[best_name][key] += group_size

    return split_records
