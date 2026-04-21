import torch
import torch.nn.functional as F
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

class QuantileHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_quantiles: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_quantiles)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The first quantile is predicted directly, and the others are predicted as positive offsets from it, to ensure monotonicity.

        raw = self.mlp(x)  # (batch, num_quantiles)

        base = raw[:, :1]  # first quantile
        deltas = F.softplus(raw[:, 1:])  # strictly positive

        quantiles = torch.cat(
            [base, base + torch.cumsum(deltas, dim=1)],
            dim=1
        )

        return quantiles
        
class MultitaskMLPWithQRegression(nn.Module):
    def __init__(
        self,
        spectrum_len: int = 1000,
        hidden_dims: list[int] = [512, 256, 128],
        num_binary: int = 166,
        categorical_sizes: dict[str, int] | None = None,
        num_regression: int = 4,
        num_quantiles: int = 3,
        dropout: float = 0.3,
        regression_hidden_dim: int = 64
    ):
        super().__init__()

        if categorical_sizes is None:
            raise ValueError("categorical_sizes is required. Use categorical_sizes_from_df() to derive it from your feature DataFrame.")

        self.categorical_names = list(categorical_sizes.keys())
        self.num_binary = num_binary
        self.categorical_sizes = categorical_sizes
        self.num_regression = num_regression
        self.num_quantiles = num_quantiles

        # Shared backbone
        layers = []
        prev = spectrum_len
        for dim in hidden_dims:
            layers.append(_mlp_block(prev, dim, dropout))
            prev = dim
        self.backbone = nn.Sequential(*layers)

        # Binary head (MACCS keys)
        self.binary_head = nn.Linear(prev, num_binary)

        # Categorical heads (one per categorical feature)
        self.categorical_heads = nn.ModuleList([
            nn.Linear(prev, num_classes)
            for num_classes in categorical_sizes.values()
        ])

        # Quantile regression head (one per element count)
        # Element counts are unique because they're the only features that we cannot easily constrain to an interval in ASP,
        # since the molecular formula is an input to the program.
        self.quantile_heads = nn.ModuleList([
            QuantileHead(prev, regression_hidden_dim, num_quantiles)
            for _ in range(num_regression)
        ])


    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "binary":      self.binary_head(h),
            "categorical": [head(h) for head in self.categorical_heads],
            "quantile": [head(h) for head in self.quantile_heads],
        }

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
        self.num_binary = num_binary
        self.categorical_sizes = categorical_sizes

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

class MultiQuantileLoss(nn.Module):
    def __init__(self, quantiles: list[float], device: torch.device = torch.device('cpu')):
        super().__init__()
        self.register_buffer('quantiles', torch.tensor(quantiles).view(1, -1).to(device))
        # shape: (1, num_quantiles)

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # preds: (batch, num_quantiles)
        # target: (batch,)
        target = target.unsqueeze(1)  # shape: (batch, 1)

        error = target - preds # shape: (batch, num_quantiles)

        loss = torch.maximum(
            self.quantiles * error,
            (self.quantiles - 1) * error
        )

        return loss.mean()  # shape: scalar