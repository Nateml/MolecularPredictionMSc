from config import config
import torch
from pathlib import Path
from data.datasets import RawFeatureDataset, dataset_from_schema
from model.model import MultitaskMLPWithQRegression, MultitaskMLP


def load_model(checkpoint_path: Path):
    """Return (model, dataset, device).

    The dataset is constructed purely for its schema attributes (maccs_length,
    feature orders, edges) — no records are loaded from disk.
    Applies saved continuous normalization stats to the dataset when present.
    """
    data_path = config.data.resolved_dataset_path()
    schema_path = config.data.resolved_schema_path()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = dataset_from_schema(data_path, schema_path,
                                  spectrum_len=config.model.spectrum_length)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Build model from checkpoint metadata so architecture matches exactly
    # regardless of current config.yaml values.
    if isinstance(dataset, RawFeatureDataset):
        model = MultitaskMLPWithQRegression(
            spectrum_len=config.model.spectrum_length,
            hidden_dims=config.model.hidden_dims,
            num_binary=ckpt.get("maccs_length", dataset.maccs_length),
            categorical_sizes=ckpt.get("categorical_sizes", dataset.categorical_sizes()),
            num_quantiles=ckpt.get("num_quantiles", config.model.num_quantiles),
            num_regression=ckpt.get("num_regression", len(dataset.continuous_feature_order)),
            regression_hidden_dim=config.model.regression_hidden_dim,
            dropout=config.model.dropout,
        )
        # Restore normalization so quantile outputs can be inverted
        # Fall back to legacy "mean"/"std" keys for older checkpoints
        cont_mean = ckpt.get("continuous_mean") or ckpt.get("mean")
        cont_std = ckpt.get("continuous_std") or ckpt.get("std")
        if cont_mean is not None and cont_std is not None:
            dataset.normalise_continuous(cont_mean, cont_std)
    else:
        model = MultitaskMLP(
            spectrum_len=config.model.spectrum_length,
            hidden_dims=config.model.hidden_dims,
            num_binary=ckpt.get("maccs_length", dataset.maccs_length),
            categorical_sizes=ckpt.get("categorical_sizes", dataset.categorical_sizes()),
            dropout=config.model.dropout,
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, dataset, device


def load_thresholds(checkpoint_path: Path) -> dict | None:
    """Load conformal thresholds saved alongside a checkpoint, or None if absent."""
    path = checkpoint_path.parent / "conformal_thresholds.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return None

