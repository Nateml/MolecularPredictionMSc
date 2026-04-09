import torch
import torch.nn as nn

"""
Multitask MLP predicting molecular features from an EI-MS spectrum.
Architecture:
    spectrum → shared backbone → binary head (MACCS keys / other binary features)
Inputs:
    x : (batch, spectrum_len)  normalised intensity vector
Outputs (dict of raw logits - sigmoid/softmax applied outside):
    "binary" : (batch, num_binary)
"""

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
        num_binary: int = 166,       # MACCS keys
        dropout: float = 0.3,
    ):
        super().__init__()

        # Shared backbone 
        layers = []
        prev = spectrum_len
        for dim in hidden_dims:
            layers.append(_mlp_block(prev, dim, dropout))
            prev = dim
        self.backbone = nn.Sequential(*layers)

        # Task-specific heads
        self.binary_head = nn.Linear(prev, num_binary)


        # self.continuous_head = nn.Linear(prev, num_continuous)
        # self.categorical_heads = nn.ModuleList([
        #     nn.Linear(prev, num_classes) for num_classes in categorical_sizes
        # ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(x)
        return {
            "binary": self.binary_head(h),
            # "continuous": self.continuous_head(h),
            # "categorical": [head(h) for head in self.categorical_heads],
        }
