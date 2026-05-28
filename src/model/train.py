from collections import defaultdict
import math
import random
from pathlib import Path
from typing import Optional
import hashlib
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from model.model import MultitaskMLP, categorical_sizes_from_df, MultiQuantileLoss
from loguru import logger

from model.logging import create_progress_bar, update_progress_bar

# Loss

def compute_loss(
    outputs: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    binary_loss_fn: nn.Module,
    categorical_loss_fns: list[nn.Module],
    quantile_loss_fn: nn.Module,
    binary_weight: float = 1.0,
    categorical_weight: float = 1.0,
    quantile_weight: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,
    return_components: bool = False
) -> torch.Tensor:

    # Binary loss (MACCS keys)
    binary_loss = binary_loss_fn(outputs["binary"], targets["binary"])
    loss = binary_weight * binary_loss

    # Categorical loss (atom counts), one loss module per head
    ce_loss = 0.0
    for i, logits in enumerate(outputs["categorical"]):
        ce_loss += categorical_loss_fns[i](logits, targets["categorical"][i])
    ce_loss /= len(outputs["categorical"])  # Average over heads

    loss += categorical_weight * ce_loss

    # Quantile loss
    quantile_loss = 0.0
    if outputs.get('quantile'):
        for i, preds in enumerate(outputs['quantile']):
            quantile_loss += quantile_loss_fn(preds, targets['regression'][:, i])
        quantile_loss /= len(outputs['quantile'])

    loss += quantile_weight * quantile_loss

    if return_components:
        return loss, binary_loss, ce_loss, quantile_loss

    return loss


# Per-epoch steps

def train_one_epoch(
    model: MultitaskMLP,
    loader: DataLoader,
    binary_loss_fn: nn.Module,
    categorical_loss_fns: list[nn.Module],
    quantile_loss_fn: nn.Module,
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
            "regression": targets["continuous"].to(device),
        }

        optimizer.zero_grad()
        outputs = model(spectra)
        loss = compute_loss(outputs, targets, binary_loss_fn, categorical_loss_fns, quantile_loss_fn, pos_weight=pos_weight)
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
    categorical_loss_fns: list[nn.Module],
    quantile_loss_fn: nn.Module,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict[str, float]:

    model.eval()
    total_loss = 0.0
    all_binary_probs,  all_binary_labels  = [], []
    all_cat_preds,     all_cat_labels     = [], []

    quantile_loss = binary_loss = ce_loss = 0.0

    for spectra, targets in loader:
        spectra = spectra.to(device)
        targets = {
            "binary":      targets["binary"].to(device),
            "categorical": [t.to(device) for t in targets["categorical"]],
            "regression": targets["continuous"].to(device),
        }

        outputs = model(spectra)
        losses = compute_loss(outputs, targets, binary_loss_fn, categorical_loss_fns, quantile_loss_fn, pos_weight=pos_weight, return_components=True)
        total_loss += losses[0].item() * len(spectra)
        binary_loss += losses[1].item() * len(spectra)
        ce_loss += losses[2].item() * len(spectra)
        quantile_loss += losses[3].item() * len(spectra)

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
        "binary_loss": binary_loss / len(loader.dataset),
        "categorical_loss": ce_loss / len(loader.dataset),
        "quantile_loss": quantile_loss / len(loader.dataset),
    }


# Main training loop

def compute_class_counts(dataset: Dataset) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Compute per-class counts for binary and categorical targets.

    Returns:
      - binary_counts: Tensor shape (num_binary,)
      - cat_counts: list of tensors, one per categorical head
    """
    # Return cached in-memory result if available
    cached = getattr(dataset, "_cached_class_counts", None)
    if cached is not None:
        return cached

    # Helper: compute a cache path when dataset exposes a data_path and _index
    def _cache_path_for(ds: Dataset) -> Optional[Path]:
        dp = getattr(ds, "data_path", None)
        idx = getattr(ds, "_index", None)
        if dp is None:
            return None
        try:
            base = Path(dp)
        except Exception:
            return None
        # Hash the index layout so subsets get distinct caches
        h = hashlib.sha1()
        if idx is not None:
            for start, end in idx:
                h.update(str(start).encode("utf-8"))
                h.update(b";")
        else:
            h.update(b"full")
        return base.with_name(base.name + f".class_counts.{h.hexdigest()}.json")

    # Try loading persistent cache
    cache_path = _cache_path_for(dataset)
    if cache_path is not None and cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            binary_counts = torch.tensor(payload["binary_counts"], dtype=torch.long)
            cat_counts = [torch.tensor(lst, dtype=torch.long) for lst in payload["cat_counts"]]
            dataset._cached_class_counts = (binary_counts, cat_counts)
            logger.info(f"Loaded class counts from cache: {cache_path}")
            return binary_counts, cat_counts
        except Exception:
            logger.info(f"Failed to load class counts cache {cache_path}, recomputing...")

    # Initialize by peeking at the first item to get shapes
    n = len(dataset)
    if n == 0:
        raise ValueError("Empty dataset")

    # Find first item
    first = dataset[0]
    # Items may be (x, targets) or (smiles, targets)
    if isinstance(first, tuple):
        _, targets = first
    else:
        raise ValueError("Unexpected dataset item format")

    binary_len = targets["binary"].shape[0]
    binary_counts = torch.zeros(binary_len, dtype=torch.long)

    # Ensure binary_counts aligns with model if provided (pad/trim later in train())

    # Initialize categorical counts as small tensors and extend as needed
    cat_counts = [torch.zeros(1, dtype=torch.long) for _ in targets["categorical"]]
    count = 0
    logger.info("Starting class count loop over {} samples", n)
    for i in range(len(dataset)):
        _, t = dataset[i]
        yb = t["binary"].long()
        binary_counts += yb
        for j, ct in enumerate(t["categorical"]):
            idx = int(ct.item())
            if idx >= len(cat_counts[j]):
                new = torch.zeros(idx + 1, dtype=torch.long)
                new[: len(cat_counts[j])] = cat_counts[j]
                cat_counts[j] = new
            cat_counts[j][idx] += 1
        count += 1
        if count % 1000 == 0:
            logger.info("Processed {}/{} samples", count, n)

    # Persist to dataset instance and optional on-disk cache
    dataset._cached_class_counts = (binary_counts, cat_counts)
    if cache_path is not None:
        try:
            payload = {
                "binary_counts": binary_counts.tolist(),
                "cat_counts": [c.tolist() for c in cat_counts],
            }
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            logger.info(f"Wrote class counts cache to {cache_path}")
        except Exception:
            logger.info(f"Failed to write class counts cache to {cache_path}")

    return binary_counts, cat_counts


def effective_num_weights_from_counts(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    # Avoid division by zero for count=0 by clamping
    counts_f = counts.to(torch.float32)
    EN = 1.0 - beta ** counts_f
    EN = torch.where(EN <= 0, torch.tensor(1.0, device=counts_f.device), EN)
    weights = (1.0 - beta) / EN
    return weights


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (batch, n_classes) for categorical or (batch, n_binary) for binary
        # targets: float tensor of same shape for binary (0/1) or long for categorical
        if logits.dim() == 2 and targets.dtype == torch.float32:
            # multi-label binary
            prob = torch.sigmoid(logits)
            targets_f = targets
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets_f, reduction='none')
            p_t = prob * targets_f + (1 - prob) * (1 - targets_f)
            focal_factor = (1 - p_t) ** self.gamma
            if self.alpha is not None:
                alpha = self.alpha.unsqueeze(0)
                loss = alpha * focal_factor * bce
            else:
                loss = focal_factor * bce
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            # assume categorical CrossEntropy case
            if self.alpha is not None:
                ce = torch.nn.functional.cross_entropy(logits, targets, weight=self.alpha, reduction=self.reduction)
            else:
                ce = torch.nn.functional.cross_entropy(logits, targets, reduction=self.reduction)
            return ce


class ZeroLoss(nn.Module):
    """A no-op loss used for constant categorical heads (single-class)."""
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Return a zero scalar on the same device as logits
        return torch.zeros((), dtype=torch.float32, device=logits.device)


def train(
    model: MultitaskMLP,
    train_dataset: Dataset,
    val_dataset: Dataset,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    pos_weight: Optional[torch.Tensor] = None,
    quantiles: list[float] = [0.05, 0.5, 0.95],
    output_path: Optional[Path] = None,
    continuous_stats: Optional[dict] = None,
) -> MultitaskMLP:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting training for {n_epochs} epochs with batch size {batch_size} and learning rate {lr:.1e}")
    logger.info(f"Using device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Compute class counts and effective-num weights
    logger.info("Computing class frequencies for class-balanced weights...")
    binary_counts, cat_counts = compute_class_counts(train_dataset)
    # binary alpha weights for positives
    beta = 0.9999
    # Ensure binary counts match model expectation
    if hasattr(model, "num_binary") and len(binary_counts) != model.num_binary:
        expected_b = model.num_binary
        if len(binary_counts) < expected_b:
            newb = torch.zeros(expected_b, dtype=torch.long)
            newb[: len(binary_counts)] = binary_counts
            binary_counts = newb
            logger.info("Padded binary counts: {} -> {}", len(binary_counts), expected_b)
        else:
            binary_counts = binary_counts[:expected_b]
            logger.info("Trimmed binary counts to {}", expected_b)

    # Compute class-balanced weights on CPU first
    binary_alpha_cpu = effective_num_weights_from_counts(binary_counts, beta=beta)

    # per-head categorical weights (create loss functions on device later)
    # Ensure counts align with model's expected categorical sizes
    expected_sizes = list(model.categorical_sizes.values()) if hasattr(model, "categorical_sizes") else [len(c) for c in cat_counts]
    if len(expected_sizes) != len(cat_counts):
        logger.info("Categorical head count mismatch: expected {} heads, found {} in counts", len(expected_sizes), len(cat_counts))

    categorical_weights_cpu = []
    for j, counts in enumerate(cat_counts):
        expected_n = expected_sizes[j] if j < len(expected_sizes) else len(counts)
        if len(counts) < expected_n:
            # pad unseen classes with zeros
            padded = torch.zeros(expected_n, dtype=torch.long)
            padded[: len(counts)] = counts
            counts_for_weights = padded
            logger.info("Padded categorical counts for head {}: {} -> {}", j, len(counts), expected_n)
        elif len(counts) > expected_n:
            # trim unexpected extra classes
            counts_for_weights = counts[:expected_n]
            logger.info("Trimmed categorical counts for head {}: {} -> {}", j, len(counts), expected_n)
        else:
            counts_for_weights = counts
        categorical_weights_cpu.append(effective_num_weights_from_counts(counts_for_weights, beta=beta))

    # Create sample weights for WeightedRandomSampler: use sum of positive-class alphas per sample (do on CPU)
    logger.info("Computing per-sample weights for WeightedRandomSampler...")
    sample_weights = []
    for i in range(len(train_dataset)):
        item = train_dataset[i]
        _, t = item
        yb = t["binary"]
        sw = float((binary_alpha_cpu * yb).sum().item())
        sample_weights.append(sw + 1.0)  # ensure non-zero

    # normalize to mean 1
    mean_sw = sum(sample_weights) / len(sample_weights)
    sample_weights = [w / mean_sw for w in sample_weights]

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Move weights to device and create loss functions
    binary_alpha = binary_alpha_cpu.to(device)

    categorical_loss_fns: list[nn.Module] = []
    expected_sizes = list(model.categorical_sizes.values()) if hasattr(model, "categorical_sizes") else None
    for j, w_cpu in enumerate(categorical_weights_cpu):
        expected_n = expected_sizes[j] if expected_sizes is not None and j < len(expected_sizes) else None
        if expected_n == 1:
            # Single-class head — skip loss computation
            logger.info("Skipping loss for categorical head {} (single class)", j)
            categorical_loss_fns.append(ZeroLoss())
        else:
            categorical_loss_fns.append(nn.CrossEntropyLoss(weight=w_cpu.to(device)))

    # Loss functions
    binary_loss_fn = ClassBalancedFocalLoss(alpha=binary_alpha, gamma=2.0)
    quantile_loss_fn = MultiQuantileLoss(quantiles=quantiles, device=device)

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

        train_loss = train_one_epoch(model, train_loader, binary_loss_fn, categorical_loss_fns, quantile_loss_fn, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, binary_loss_fn, categorical_loss_fns, quantile_loss_fn, device, pos_weight)

        # Save latest checkpoint
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "categorical_sizes": model.categorical_sizes,
            "maccs_length": model.num_binary,
            "num_regression": model.num_regression,
            "num_quantiles": model.num_quantiles,
            **(continuous_stats or {}),
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
                "num_regression": model.num_regression,
                "num_quantiles": model.num_quantiles,
                "maccs_length": model.num_binary,
                **(continuous_stats or {}),
            }
            torch.save(best_ckpt, output_path.with_suffix('.best.pt'))

        logger.info(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_binary_acc={val_metrics['binary_accuracy']:.4f} | "
            f"val_cat_acc={val_metrics['categorical_accuracy']:.4f} | "
            f"val_quantile_loss={val_metrics['quantile_loss']:.4f} | "
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
