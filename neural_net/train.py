import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from typing import Optional

from neural_net.model import MultitaskMLP
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
    binary_weight: float = 1.0,
    pos_weight: Optional[torch.Tensor] = None,   # class-imbalance correction
) -> torch.Tensor:
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = binary_weight * loss_fn(outputs["binary"], targets["binary"])
    return loss


# Per-epoch steps

def train_one_epoch(
    model: MultitaskMLP,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> float:
    model.train()
    total_loss = 0.0

    for spectra, targets in loader:
        spectra = spectra.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        outputs = model(spectra)
        loss = compute_loss(outputs, targets, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(spectra)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: MultitaskMLP,
    loader: DataLoader,
    device: torch.device,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict[str, float]:

    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for spectra, targets in loader:
        spectra = spectra.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model(spectra)
        total_loss += compute_loss(outputs, targets, pos_weight=pos_weight).item() * len(spectra)

        all_probs.append(torch.sigmoid(outputs["binary"]).cpu())
        all_labels.append(targets["binary"].cpu())

    all_probs  = torch.cat(all_probs)   
    all_labels = torch.cat(all_labels)  

    preds    = (all_probs >= 0.5).float()
    accuracy = (preds == all_labels).float().mean().item()

    return {
        "loss":            total_loss / len(loader.dataset),
        "binary_accuracy": accuracy,
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

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight)
        val_metrics = evaluate(model, val_loader, device, pos_weight)
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['binary_accuracy']:.4f}"
        )

    return model


# Entry point 
# Run once the data pipeline is ready
#
#   train_ds, val_ds, test_ds = split_dataset(dataset)
#
#   model = MultitaskMLP()
#   train(model, train_ds, val_ds)
#
#   test_metrics = evaluate(model, DataLoader(test_ds, batch_size=64), device)
