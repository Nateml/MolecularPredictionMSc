import torch
import torch.nn as nn
import pandas as pd

"""
Multitask MLP predicting molecular features from an EI-MS spectrum.

Architecture:
    spectrum → shared backbone → binary head      (MACCS keys)
                               → categorical heads (numerical features as classes)

Inputs:
    x : (batch, spectrum_len)  normalised intensity vector

Outputs (dict of raw logits — sigmoid/softmax applied outside):
    "binary"      : (batch, num_binary)
    "categorical" : list of (batch, num_classes), one tensor per categorical feature

Build from data (no hardcoded feature lists):
    categorical_sizes = categorical_sizes_from_df(feature_df, binary_cols)
    model = MultitaskMLP(spectrum_len=..., num_binary=len(binary_cols), categorical_sizes=categorical_sizes)
"""


def categorical_sizes_from_df(df: pd.DataFrame, binary_cols: list[str]) -> dict[str, int]:
    numerical_cols = [c for c in df.columns if c not in binary_cols]
    return {col: int(df[col].max()) + 1 for col in numerical_cols}


def _mlp_block(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class MultitaskMLP(nn.Module):
    def __init__(
        self,
        spectrum_len: int = 1000,
        hidden_dims: list[int] = [512, 256, 128],
        num_binary: int = 166,
        categorical_sizes: dict[str, int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()

        if categorical_sizes is None:
            raise ValueError("categorical_sizes is required. Use categorical_sizes_from_df() to derive it from your feature DataFrame.")

        self.categorical_names = list(categorical_sizes.keys())

        # Shared backbone
        layers = []
        prev = spectrum_len
        for dim in hidden_dims:
            layers.append(_mlp_block(prev, dim, dropout))
            prev = dim
        self.backbone = nn.Sequential(*layers)

        # Binary head (MACCS keys)
        self.binary_head = nn.Linear(prev, num_binary)

        # Categorical heads (one per atom count feature)
        self.categorical_heads = nn.ModuleList([
            nn.Linear(prev, num_classes)
            for num_classes in categorical_sizes.values()
        ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "binary":      self.binary_head(h),
            "categorical": [head(h) for head in self.categorical_heads],
        }
