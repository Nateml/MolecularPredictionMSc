import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _conv1d_block(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int | None = None, dropout: float = 0.2) -> nn.Sequential:
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Dropout(dropout),
    )


class MultitaskCNN(nn.Module):
    """Multitask model with a 1D convolutional backbone.

    Inputs:
        x : (batch, spectrum_len)  -- single-channel spectrum

    Internals:
        - Unsqueeze channel dim to (batch, 1, spectrum_len)
        - Conv1D layers: 64(k=5) → 128(k=5) → 128(k=3) → MaxPool → 256(k=3) → 256(k=3)
        - Global max pooling to fixed-size vector
        - Dense projection layers
        - Same heads as existing MLP models
    """
    def __init__(
        self,
        spectrum_len: int = 1000,
        num_binary: int = 166,
        categorical_sizes: dict[str, int] | None = None,
        num_regression: int = 0,
        num_quantiles: int = 3,
        regression_hidden_dim: int = 64,
        dropout: float = 0.3,
        global_feat_dim: int = 128,
    ):
        super().__init__()

        if categorical_sizes is None:
            raise ValueError("categorical_sizes is required. Use categorical_sizes_from_df() to derive it from your feature DataFrame.")

        self.categorical_names = list(categorical_sizes.keys())
        self.num_binary = num_binary
        self.num_regression = num_regression
        self.num_quantiles = num_quantiles
        self.categorical_sizes = categorical_sizes

        # Build convolutional backbone with specific architecture:
        # Conv1D 64 (k=5) → Conv1D 128 (k=5) → Conv1D 128 (k=3) → MaxPool2 →
        # Conv1D 256 (k=3) → Conv1D 256 (k=3) → GlobalMaxPool
        self.conv_backbone = nn.Sequential(
            _conv1d_block(1, 64, kernel_size=5, dropout=dropout),
            _conv1d_block(64, 128, kernel_size=5, dropout=dropout),
            _conv1d_block(128, 128, kernel_size=3, dropout=dropout),
            nn.MaxPool1d(kernel_size=2, stride=2),
            _conv1d_block(128, 256, kernel_size=3, dropout=dropout),
            _conv1d_block(256, 256, kernel_size=3, dropout=dropout),
        )

        # Global max pooling to fixed-size vector
        self.global_pool = nn.AdaptiveMaxPool1d(1)  # -> (batch, 256, 1)

        # Final projection: 256 → 256 (ReLU) → Dropout → 128
        self.proj = nn.Sequential(
            nn.Flatten(),  # from (batch, 256, 1) to (batch, 256)
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, global_feat_dim),
            nn.ReLU(),
        )

        # Heads (same as MLP)
        self.binary_head = nn.Linear(global_feat_dim, num_binary)
        self.categorical_heads = nn.ModuleList([
            nn.Linear(global_feat_dim, num_classes)
            for num_classes in categorical_sizes.values()
        ])

        # Optional quantile regression heads (one per continuous target)
        if self.num_regression and self.num_regression > 0:
            self.quantile_heads = nn.ModuleList([
                QuantileHead(global_feat_dim, regression_hidden_dim, num_quantiles)
                for _ in range(self.num_regression)
            ])

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Input expected (batch, spectrum_len) -> make (batch, 1, spectrum_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        h = self.conv_backbone(x)
        h = self.global_pool(h)  # (batch, 256, 1)
        feat = self.proj(h)      # (batch, global_feat_dim)

        out = {
            "binary":      self.binary_head(feat),
            "categorical": [head(feat) for head in self.categorical_heads],
        }

        if getattr(self, "num_regression", 0) and getattr(self, "quantile_heads", None) is not None:
            out["quantile"] = [head(feat) for head in self.quantile_heads]

        return out


class MultitaskCNNWithDreams(nn.Module):
    """Multitask model with DreaMS peak-list transformer encoder backbone.
    
    Replaces the CNN with a transformer encoder operating on variable-length
    (m/z, intensity) peak lists. Keeps the same multitask heads as the original
    MultitaskCNN model.
    
    Supports loading pretrained DreaMS encoder weights.
    
    Inputs:
        Can accept either:
        1. Dense spectrum vectors: (batch, spectrum_len) → internally converted to peak lists
        2. Peak list tensors directly: (batch, max_peaks, 2)
        
        Plus optional precursor_mz: (batch,) values
    
    Outputs (dict of raw logits):
        "binary"      : (batch, num_binary)
        "categorical" : list of (batch, num_classes), one tensor per categorical feature
        "quantile"    : (optional) list of (batch, num_quantiles), one per regression target
    """
    
    def __init__(
        self,
        spectrum_len: int = 1000,
        mz_min: float = 0.0,
        mz_max: float = 1000.0,
        num_binary: int = 166,
        categorical_sizes: dict[str, int] | None = None,
        num_regression: int = 0,
        num_quantiles: int = 3,
        regression_hidden_dim: int = 64,
        dropout: float = 0.2,
        # DreaMS encoder parameters
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        ff_dim: int = 1024,
        use_precursor_token: bool = True,
        pretrained_encoder_path: str | None = None,
    ):
        super().__init__()
        
        if categorical_sizes is None:
            raise ValueError("categorical_sizes is required. Use categorical_sizes_from_df() to derive it from your feature DataFrame.")
        
        self.categorical_names = list(categorical_sizes.keys())
        self.num_binary = num_binary
        self.num_regression = num_regression
        self.num_quantiles = num_quantiles
        self.categorical_sizes = categorical_sizes
        self.spectrum_len = spectrum_len
        self.mz_min = mz_min
        self.mz_max = mz_max
        
        # Import DreaMS encoder
        from .dreams_encoder import DreaMSEncoder, load_pretrained_dreams
        
        # Build DreaMS encoder
        self.encoder = DreaMSEncoder(
            hidden_dim=hidden_dim,
            num_transformer_layers=num_transformer_layers,
            num_attention_heads=num_attention_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            use_precursor_token=use_precursor_token,
        )
        
        # Load pretrained weights if provided
        if pretrained_encoder_path is not None:
            self._load_pretrained(pretrained_encoder_path)
        
        # Multitask heads (same as original MultitaskCNN)
        self.binary_head = nn.Linear(hidden_dim, num_binary)
        self.categorical_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes)
            for num_classes in categorical_sizes.values()
        ])
        
        if self.num_regression and self.num_regression > 0:
            self.quantile_heads = nn.ModuleList([
                QuantileHead(hidden_dim, regression_hidden_dim, num_quantiles)
                for _ in range(self.num_regression)
            ])
    
    def _load_pretrained(self, path: str | None) -> None:
        """Load pretrained encoder weights."""
        if path is None:
            return
        
        try:
            from .dreams_encoder import load_pretrained_dreams
            pretrained_encoder = load_pretrained_dreams(path)
            self.encoder.load_state_dict(pretrained_encoder.state_dict())
            print(f"Loaded pretrained DreaMS encoder from {path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained encoder from {path}: {e}")
    
    def _dense_spectrum_to_peaks(self, spectrum_dense: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert dense spectrum vector to approximate peak list.
        
        Args:
            spectrum_dense: (batch, spectrum_len)
        
        Returns:
            peaks: (batch, max_peaks, 2) - padded peak lists
            mask: (batch, max_peaks) - attention mask
        """
        from .dreams_encoder import pad_peak_batch
        
        batch_peak_lists = []
        
        for spec in spectrum_dense:
            # Find peaks (local maxima and non-zero values)
            peaks_idx = torch.nonzero(spec > 0, as_tuple=True)[0]
            
            if len(peaks_idx) == 0:
                # No peaks found - create empty peak list
                batch_peak_lists.append(torch.zeros((0, 2), dtype=torch.float32, device=spec.device))
            else:
                # Convert bin indices to m/z values
                mz_range = self.mz_max - self.mz_min
                scale = mz_range / max(1, self.spectrum_len - 1)
                mz_values = self.mz_min + peaks_idx.float() * scale
                intensities = spec[peaks_idx]
                
                peaks = torch.stack([mz_values, intensities], dim=1)
                
                # Normalize to [0, 1]
                peaks[:, 0] = (peaks[:, 0] - self.mz_min) / max(1e-6, mz_range)
                peaks[:, 1] = peaks[:, 1] / (peaks[:, 1].max() + 1e-6)
                
                batch_peak_lists.append(peaks)
        
        return pad_peak_batch(batch_peak_lists)
    
    def forward(
        self,
        x: torch.Tensor,
        precursor_mz: torch.Tensor | None = None,
        peaks_list: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Either dense spectrum (batch, spectrum_len) or peak list (batch, max_peaks, 2)
            precursor_mz: Optional (batch,) precursor m/z values
            peaks_list: If provided, use this directly instead of x (overrides x)
            attention_mask: Optional (batch, num_peaks) boolean mask for variable-length peak lists
        
        Returns:
            Dictionary with model outputs
        """
        # Handle peak list input
        if peaks_list is not None:
            peaks = peaks_list
            if attention_mask is None:
                # Create mask: all non-zero peaks are valid
                attention_mask = (peaks[:, :, 0] != 0).any(dim=2)
        elif x.dim() == 2 and x.shape[1] == self.spectrum_len:
            # Input is dense spectrum - convert to peaks
            peaks, attention_mask = self._dense_spectrum_to_peaks(x)
        elif x.dim() == 3 and x.shape[2] == 2:
            # Input is already peak list format
            peaks = x
            if attention_mask is None:
                attention_mask = (peaks[:, :, 0] != 0).any(dim=2)
        else:
            raise ValueError(
                f"Unexpected input shape: {x.shape}. "
                f"Expected either (batch, {self.spectrum_len}) for dense spectrum "
                f"or (batch, max_peaks, 2) for peak lists."
            )
        
        # Forward through DreaMS encoder
        feat = self.encoder(peaks, precursor_mz=precursor_mz, attention_mask=attention_mask)
        
        # Apply multitask heads
        out = {
            "binary": self.binary_head(feat),
            "categorical": [head(feat) for head in self.categorical_heads],
        }
        
        if getattr(self, "num_regression", 0) and getattr(self, "quantile_heads", None) is not None:
            out["quantile"] = [head(feat) for head in self.quantile_heads]
        
        return out
    
    def freeze_encoder(self) -> None:
        """Freeze encoder weights for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
