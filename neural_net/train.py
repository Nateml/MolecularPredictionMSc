import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Optional

from neural_net.model import MultitaskMLP, categorical_sizes_from_df
from torch.utils.data import Dataset


# Split

def split_dataset(
    dataset: Dataset,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:

    n = len(dataset)
    n_train = int(n * train_fraction)
    n_val   = int(n * val_fraction)
    n_test  = n - n_train - n_val
    return random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))


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
) -> MultitaskMLP:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)

    binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
    categorical_loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, binary_loss_fn, categorical_loss_fn, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, binary_loss_fn, categorical_loss_fn, device, pos_weight)
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_binary_acc={val_metrics['binary_accuracy']:.4f} | "
            f"val_cat_acc={val_metrics['categorical_accuracy']:.4f}"
        )

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
#   train_ds, val_ds, test_ds = split_dataset(dataset)
#   train(model, train_ds, val_ds)
#   test_metrics = evaluate(model, DataLoader(test_ds, batch_size=64), device)
