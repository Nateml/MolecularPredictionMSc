# Description: Splits the dataset and saves the splits to disk
import os
import pickle
import torch

from torch.utils.data import Dataset, Subset
from collections import defaultdict
from typing import Final, Union, Optional
from neural_net.datasets import SpectrumDataset

# Constants for dataset splitting
TRAIN_FRACTION: Final[float] = 0.75
VAL_FRACTION: Final[float] = 0.10
CALIBRATION_FRACTION: Final[float] = 0.05
TEST_FRACTION: Final[float] = 0.10

# Set project root for path resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def split_dataset(
    dataset: SpectrumDataset,
    train_fraction: float = TRAIN_FRACTION,
    val_fraction: float = VAL_FRACTION,
    calibration_fraction: float = CALIBRATION_FRACTION,
    test_fraction: float = TEST_FRACTION,
    seed: int = 42,
    check_label_consistency: bool = True,
    output_dir: Optional[str] = "../data/splits",
) -> tuple[SpectrumDataset, SpectrumDataset, SpectrumDataset, SpectrumDataset]:
    """
    Splits the given dataset into train, validation, calibration, and test subsets according to the specified fractions.

    Args:
        dataset (SpectrumDataset): The full dataset to split. Must have a `records` attribute containing
            the original data records with SMILES fields for grouping.
        train_fraction (float): Fraction of the dataset to allocate to the training set.
        val_fraction (float): Fraction of the dataset to allocate to the validation set.
        calibration_fraction (float): Fraction of the dataset to allocate to the calibration set.
        test_fraction (float): Fraction of the dataset to allocate to the test set.
        seed (int): Random seed for reproducibility of the split.
        check_label_consistency (bool): If True, checks that records with identical SMILES have
            identical labels. If False, skips this check (use with caution, as it may lead to data leakage if duplicates have conflicting labels).
        output_dir (Optional[str]): If provided, saves the split indices to this directory as python pickle files.
    """

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

    subsets = {
        split_name: Subset(dataset, split_indices[split_name])
        for split_name in split_names
    }

    # Save subsets to disk
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        for name, subset in subsets.items():
            output_path = os.path.join(output_dir, f"{name}.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(subset, f)


    return tuple(
        Subset(dataset, split_indices[split_name])
        for split_name in split_names
    )


# -------------------------
# Internal helpers
# -------------------------

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
    desired_totals = {
        split_name: total_records * fraction
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
            total_score = abs(prospective_total - desired_totals[split_name])

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
        split_stratum_totals[best_split_name][strata_key] += group_size

    return split_records