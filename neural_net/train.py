from collections import defaultdict
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from neural_net.model import MultitaskMLP, categorical_sizes_from_df
from loguru import logger

from neural_net.logging import create_progress_bar, update_progress_bar


# Split

## example data entry:
#{"spectrum": , "smiles": "CCO", "structural_vector_onehot_nbits": 262, "feature_vector_onehot_nbits": 429, "structural_vector_onehot_packed_b64": "ilStLQqiiAoggqCoIxKkQVUEQaqCmVIVKRpUYKqgiqCg", "feature_vector_onehot_packed_b64": "AAAAAAAACAJAMZYPwAJE7QVHjir9FKlaWhVFEBRBBUFQRiVIgqoIg1UFMqQqUjSowVVBFUFA", "combined_nbits": 429, "combined_packed_b64": "AAAAAAAACAJAMZYPwAJE7QVHjir9FKlaWhVFEBRBBUFQRiVIgqoIg1UFMqQqUjSowVVBFUFA"},


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


def split_dataset(
    dataset: Dataset,
    train_fraction: float = 0.75,
    val_fraction: float = 0.10,
    calibration_fraction: float = 0.05,
    test_fraction: float = 0.10,
    seed: int = 42,
    check_label_consistency: bool = True,
) -> tuple[Dataset, Dataset, Dataset, Dataset]:

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

    return tuple(
        Subset(dataset, split_indices[split_name])
        for split_name in split_names
    )


# Loss

def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    binary_loss_fn: nn.Module,
    categorical_loss_fn: nn.Module,
    binary_weight: float = 1.0,
    categorical_weight: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    # Binary loss (MACCS keys)
    loss = binary_weight * binary_loss_fn(outputs["binary"], targets["binary"])

    # Categorical loss (atom counts), one CrossEntropyLoss per head
    ce_loss = 0.0
    for i, logits in enumerate(outputs["categorical"]):
        ce_loss += categorical_loss_fn(logits, targets["categorical"][i])
    ce_loss /= len(outputs["categorical"])  # Average over heads

    loss = loss + categorical_weight * ce_loss

    return loss


# Per-epoch steps

def train_one_epoch(
    model: MultitaskMLP,
    loader: DataLoader,
    binary_loss_fn: nn.Module,
    categorical_loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0

    progress_bar = create_progress_bar(loader, desc='Training')

    for spectra, targets in loader:
        spectra = spectra.to(device)
        targets = {
            "binary":      targets["binary"].to(device),
            "categorical": [t.to(device) for t in targets["categorical"]],
        }

        optimizer.zero_grad()
        outputs = model(spectra)
        loss = compute_loss(outputs, targets, binary_loss_fn, categorical_loss_fn, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        # Update progres bar
        update_progress_bar(progress_bar, loss.item())

        total_loss += loss.item() * len(spectra)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: MultitaskMLP,
    loader: DataLoader,
    binary_loss_fn: nn.Module,
    categorical_loss_fn: nn.Module,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict[str, float]:

    model.eval()
    total_loss = 0.0
    all_binary_probs,  all_binary_labels  = [], []
    all_cat_preds,     all_cat_labels     = [], []

    for spectra, targets in loader:
        spectra = spectra.to(device)
        targets = {
            "binary":      targets["binary"].to(device),
            "categorical": [t.to(device) for t in targets["categorical"]],
        }

        outputs = model(spectra)
        total_loss += compute_loss(outputs, targets, binary_loss_fn, categorical_loss_fn, pos_weight=pos_weight).item() * len(spectra)

        # Binary
        all_binary_probs.append(torch.sigmoid(outputs["binary"]).cpu())
        all_binary_labels.append(targets["binary"].cpu())

        # Categorical 
        batch_cat_preds  = [logits.argmax(dim=1).cpu() for logits in outputs["categorical"]]
        batch_cat_labels = [t.cpu() for t in targets["categorical"]]
        all_cat_preds.append(batch_cat_preds)
        all_cat_labels.append(batch_cat_labels)

    # Binary accuracy
    all_binary_probs  = torch.cat(all_binary_probs)
    all_binary_labels = torch.cat(all_binary_labels)
    binary_acc = ((all_binary_probs >= 0.5).float() == all_binary_labels).float().mean().item()

    # Categorical accuracy (average across all heads)
    n_heads = len(all_cat_preds[0])
    cat_acc = 0.0
    for i in range(n_heads):
        preds  = torch.cat([batch[i] for batch in all_cat_preds])
        labels = torch.cat([batch[i] for batch in all_cat_labels])
        cat_acc += (preds == labels).float().mean().item()
    cat_acc /= n_heads

    return {
        "loss":                total_loss / len(loader.dataset),
        "binary_accuracy":     binary_acc,
        "categorical_accuracy": cat_acc,
    }


# Main training loop

def train(
    model: MultitaskMLP,
    train_dataset: Dataset,
    val_dataset: Dataset,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    pos_weight: Optional[torch.Tensor] = None,
    output_path: Optional[Path] = None,
) -> MultitaskMLP:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting training for {n_epochs} epochs with batch size {batch_size} and learning rate {lr:.1e}")
    logger.info(f"Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    categorical_loss_fn = nn.CrossEntropyLoss()

    # Checkpoint/resume support
    # If OUTPUT_PATH exists and contains epoch, resume from next epoch.
    start_epoch = 1
    checkpoint_path_latest = output_path.with_suffix('.latest.pt') if output_path else None
    if checkpoint_path_latest and checkpoint_path_latest.exists():
        logger.info(f"Found checkpoint {checkpoint_path_latest}, attempting to resume...")
        ckpt = torch.load(checkpoint_path_latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"]) 
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        logger.info(f"Resuming from epoch {start_epoch}")

    best_val_loss = float("inf")
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, n_epochs + 1):
        logger.info(f"Epoch {epoch}/{n_epochs}--------------------------------------")

        train_loss = train_one_epoch(model, train_loader, binary_loss_fn, categorical_loss_fn, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, binary_loss_fn, categorical_loss_fn, device, pos_weight)

        # Save latest checkpoint
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "categorical_sizes": model.categorical_sizes,
            "maccs_length": model.num_binary
        }
        if checkpoint_path_latest:
            torch.save(latest_ckpt, checkpoint_path_latest)

        # Save best model seperately
        if val_metrics["loss"] < best_val_loss and output_path:
            best_val_loss = val_metrics["loss"]
            best_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "categorical_sizes": model.categorical_sizes,
                "maccs_length": model.num_binary
            }
            torch.save(best_ckpt, output_path.with_suffix('.best.pt'))

        logger.info(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_binary_acc={val_metrics['binary_accuracy']:.4f} | "
            f"val_cat_acc={val_metrics['categorical_accuracy']:.4f}"
        )
        logger.info("-----------------------------------\n")

    logger.info("Training complete.")

    return model


# Entry point — fill in once the data pipeline is ready
#
#   spectra       : 2D array/tensor of shape (n_samples, spectrum_len)
#   feature_df    : DataFrame with only feature columns (no smiles/inchikey/etc.)
#   binary_cols   : list of MACCS key column names (e.g. ["maccs_1", ..., "maccs_166"])
#   dataset       : a torch Dataset whose __getitem__ returns
#                     (spectrum_tensor, {"binary": tensor, "categorical": [tensor, ...]})
#
#   categorical_sizes = categorical_sizes_from_df(feature_df, binary_cols)
#
#   model = MultitaskMLP(
#       spectrum_len      = spectra.shape[1],
#       num_binary        = len(binary_cols),
#       categorical_sizes = categorical_sizes,
#   )
#
#   train_ds, val_ds, calibration_ds, test_ds = split_dataset(dataset)
#   train(model, train_ds, val_ds)
#   test_metrics = evaluate(model, DataLoader(test_ds, batch_size=64), device)
