from src.config import config
import torch
from pathlib import Path
from data.datasets import RawFeatureDataset, dataset_from_schema
from model.model import MultitaskMLPWithQRegression, MultitaskMLP, MultitaskCNN


def _infer_model_class(state_dict: dict) -> str:
    """Infer model class name from state dict keys for old checkpoints lacking 'model_class'."""
    keys = state_dict.keys()
    if any(k.startswith("conv_backbone.") for k in keys):
        return "MultitaskCNN"
    if any(k.startswith("quantile_heads.") for k in keys):
        return "MultitaskMLPWithQRegression"
    return "MultitaskMLP"


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

    model_class = ckpt.get("model_class") or _infer_model_class(ckpt["model_state_dict"])

    num_binary = ckpt.get("maccs_length", dataset.maccs_length)
    categorical_sizes = ckpt.get("categorical_sizes", dataset.categorical_sizes())
    num_quantiles = ckpt.get("num_quantiles", config.model.num_quantiles)
    num_regression = ckpt.get("num_regression", 0)

    if model_class == "MultitaskCNN":
        model = MultitaskCNN(
            spectrum_len=config.model.spectrum_length,
            num_binary=num_binary,
            categorical_sizes=categorical_sizes,
            num_quantiles=num_quantiles,
            num_regression=num_regression,
            regression_hidden_dim=getattr(config.model, "regression_hidden_dim", 64),
            dropout=config.model.dropout,
            global_feat_dim=getattr(config.model, "global_feat_dim", 128),
        )
    elif model_class == "MultitaskMLPWithQRegression":
        model = MultitaskMLPWithQRegression(
            spectrum_len=config.model.spectrum_length,
            hidden_dims=config.model.hidden_dims,
            num_binary=num_binary,
            categorical_sizes=categorical_sizes,
            num_quantiles=num_quantiles,
            num_regression=ckpt.get("num_regression", len(dataset.continuous_feature_order)),
            regression_hidden_dim=config.model.regression_hidden_dim,
            dropout=config.model.dropout,
        )
    else:
        model = MultitaskMLP(
            spectrum_len=config.model.spectrum_length,
            hidden_dims=config.model.hidden_dims,
            num_binary=num_binary,
            categorical_sizes=categorical_sizes,
            dropout=config.model.dropout,
        )

    # Restore normalization so quantile outputs can be inverted.
    # Fall back to legacy "mean"/"std" keys for older checkpoints.
    if isinstance(dataset, RawFeatureDataset):
        cont_mean = ckpt.get("continuous_mean") or ckpt.get("mean")
        cont_std = ckpt.get("continuous_std") or ckpt.get("std")
        if cont_mean is not None and cont_std is not None:
            dataset.normalise_continuous(cont_mean, cont_std)

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, dataset, device


def thresholds_path(checkpoint_path: Path) -> Path:
    """Return the path where conformal thresholds for a checkpoint are stored."""
    return checkpoint_path.parent / f"{checkpoint_path.stem}.thresholds.pt"


def load_thresholds(checkpoint_path: Path) -> dict | None:
    """Load conformal thresholds saved alongside a checkpoint, or None if absent."""
    path = thresholds_path(checkpoint_path)
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return None

