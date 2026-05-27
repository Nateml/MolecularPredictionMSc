"""
DreaMS-inspired Peak-List Transformer Encoder for Mass Spectrometry.

Based on the DreaMS architecture:
  - Encodes variable-length (m/z, intensity) peak lists
  - Prepends special token for precursor m/z
  - Uses transformer encoder with self-attention over peaks
  - Global average pooling to fixed-size representation
  - Can load pretrained weights from Hugging Face
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from pathlib import Path


class PeakTokenizer(nn.Module):
    """Embed (m/z, intensity) peaks into a learned vector space."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Linear projection for m/z and intensity (2 inputs → hidden_dim)
        self.peak_embed = nn.Linear(2, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, peaks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            peaks: (batch, num_peaks, 2) where last dim is [m/z, intensity]
        
        Returns:
            embedded: (batch, num_peaks, hidden_dim)
        """
        embedded = self.peak_embed(peaks)  # (batch, num_peaks, hidden_dim)
        embedded = self.dropout(embedded)
        return embedded


class DreaMSEncoder(nn.Module):
    """
    Transformer encoder for mass spectra.
    
    Takes variable-length peak lists and produces fixed-size representations
    via self-attention and global pooling.
    
    Inputs:
        - Peak lists as (m/z, intensity) pairs
        - Optional: precursor m/z value
        - Variable sequence length handled via attention masks
    
    Outputs:
        - Fixed-size embedding (batch, hidden_dim)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_transformer_layers: int = 4,
        num_attention_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = True,
        use_precursor_token: bool = True,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_precursor_token = use_precursor_token
        
        # Peak tokenizer: converts (m/z, intensity) → hidden_dim
        self.peak_tokenizer = PeakTokenizer(hidden_dim, dropout)
        
        # Optional: special token for precursor m/z
        if use_precursor_token:
            self.precursor_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
            nn.init.normal_(self.precursor_token, std=0.02)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_seq_len=2048, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
            enable_nested_tensor=False,  # For variable-length sequences
        )
        
    def forward(
        self,
        peaks: torch.Tensor,
        precursor_mz: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            peaks: (batch, num_peaks, 2) where dim 2 is [m/z, intensity]
            precursor_mz: (batch,) optional precursor mass-to-charge
            attention_mask: (batch, 1 + num_peaks) optional; True = attend, False = ignore
                           (follows PyTorch convention for transformer)
        
        Returns:
            embedding: (batch, hidden_dim) - fixed-size output from global pooling
        """
        batch_size = peaks.shape[0]
        
        # Embed peaks
        peak_embeds = self.peak_tokenizer(peaks)  # (batch, num_peaks, hidden_dim)
        
        # Prepend precursor token if provided
        if self.use_precursor_token:
            # Expand precursor token for batch
            precursor_tokens = self.precursor_token.expand(batch_size, -1, -1)  # (batch, 1, hidden_dim)
            
            # If precursor_mz provided, embed it into the special token
            if precursor_mz is not None:
                # Normalize precursor_mz to be in reasonable range (assuming [0, 1])
                precursor_scaled = precursor_mz.view(batch_size, 1)
                precursor_embed = torch.cat([
                    precursor_scaled,
                    torch.ones_like(precursor_scaled)  # Set intensity to 1 for special token
                ], dim=1).unsqueeze(1)  # (batch, 1, 2)
                precursor_embed = self.peak_tokenizer(precursor_embed)  # (batch, 1, hidden_dim)
                precursor_tokens = precursor_embed
            
            # Concatenate precursor token with peaks
            sequence = torch.cat([precursor_tokens, peak_embeds], dim=1)  # (batch, 1 + num_peaks, hidden_dim)
            
            # Update attention mask if provided
            if attention_mask is not None:
                # attention_mask for peaks should already exclude padding
                # Prepend True for precursor token
                precursor_mask = torch.ones(batch_size, 1, device=attention_mask.device, dtype=torch.bool)
                attention_mask = torch.cat([precursor_mask, attention_mask], dim=1)
        else:
            sequence = peak_embeds
        
        # Add positional encodings
        sequence = self.pos_encoder(sequence)
        
        # Apply transformer encoder
        # PyTorch transformer expects attention_mask with shape (batch, seq_len) or (batch, seq_len, seq_len)
        # and True values are KEPT (not masked out like in some frameworks)
        if attention_mask is not None:
            # Invert if needed: PyTorch keeps True values, ignores False
            attention_mask_pt = attention_mask.to(torch.bool)
            output = self.transformer_encoder(sequence, src_key_padding_mask=~attention_mask_pt)
        else:
            output = self.transformer_encoder(sequence)
        
        # Global average pooling over sequence dimension
        # We need to mask out padding positions before averaging
        if attention_mask is not None:
            # Expand mask for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)
            masked_output = output * mask_expanded  # Zero out padding positions
            pooled = masked_output.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)  # (batch, hidden_dim)
        else:
            pooled = output.mean(dim=1)  # (batch, hidden_dim)
        
        return pooled


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_seq_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create position indices
        position = torch.arange(max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.shape[1]
        return x + self.pe[:, :seq_len, :] + self.dropout(x - x)  # Dropout applied


def load_pretrained_dreams(
    model_name: str = "NateML/DreaMS-pretrained",
    cache_dir: Optional[str] = None,
    device: Optional[str] = None,
) -> DreaMSEncoder:
    """
    Load a pretrained DreaMS encoder from Hugging Face Hub.
    
    Args:
        model_name: Hugging Face model identifier (e.g., "NateML/DreaMS-pretrained")
        cache_dir: Directory to cache downloaded model weights
        device: Device to load model on ('cuda', 'cpu', etc.)
    
    Returns:
        DreaMSEncoder with loaded pretrained weights
    
    Note:
        As of my knowledge cutoff, DreaMS may not have official HF weights yet.
        You can:
        1. Save your own pretrained encoder and reference it
        2. Use the model ID if/when official weights are released
        3. Provide a local path to saved weights instead
    """
    try:
        from transformers import AutoModel
        
        # Try to load from Hugging Face Hub
        model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
        return model
    except (ImportError, OSError) as e:
        # Fallback: load from local path if provided
        if Path(model_name).exists():
            checkpoint = torch.load(model_name, map_location=device or "cpu")
            encoder = DreaMSEncoder()
            if "state_dict" in checkpoint:
                encoder.load_state_dict(checkpoint["state_dict"])
            else:
                encoder.load_state_dict(checkpoint)
            return encoder
        else:
            raise ValueError(
                f"Could not load DreaMS from '{model_name}'. "
                f"Please provide either:\n"
                f"  1. A valid Hugging Face model ID (requires internet)\n"
                f"  2. A local path to a saved checkpoint"
            ) from e


# ============================================================================
# Utility function to convert spectrum string to peak list
# ============================================================================

def spectrum_string_to_peaks(
    spectrum: str,
    mz_min: float = 0.0,
    mz_max: float = 1000.0,
    normalize_intensity: bool = True,
    precursor_mz: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
    """
    Convert a spectrum string to peak tensors.
    
    Args:
        spectrum: Space-separated "mz:intensity" pairs
        mz_min: Minimum m/z threshold
        mz_max: Maximum m/z threshold
        normalize_intensity: Whether to normalize intensities to [0, 1]
        precursor_mz: Optional precursor m/z value
    
    Returns:
        peaks: (num_peaks, 2) tensor of [m/z, intensity]
        precursor_tensor: (1,) tensor of precursor m/z or None
        mask: (num_peaks,) boolean mask of valid peaks
    """
    peaks_list = []
    
    if spectrum:
        for token in spectrum.strip().split():
            try:
                mz_str, intensity_str = token.split(":")
                mz = float(mz_str)
                intensity = float(intensity_str)
            except ValueError:
                continue
            
            if mz < mz_min or mz > mz_max:
                continue
            
            peaks_list.append([mz, intensity])
    
    if peaks_list:
        peaks = torch.tensor(peaks_list, dtype=torch.float32)
        
        # Normalize m/z to [0, 1]
        mz_range = mz_max - mz_min
        if mz_range > 0:
            peaks[:, 0] = (peaks[:, 0] - mz_min) / mz_range
        
        # Normalize intensities to [0, 1]
        if normalize_intensity:
            max_intensity = peaks[:, 1].max()
            if max_intensity > 0:
                peaks[:, 1] = peaks[:, 1] / max_intensity
    else:
        peaks = torch.zeros((0, 2), dtype=torch.float32)
    
    # Create attention mask (all valid peaks)
    mask = torch.ones(peaks.shape[0], dtype=torch.bool)
    
    # Precursor m/z tensor
    precursor_tensor = None
    if precursor_mz is not None:
        precursor_tensor = torch.tensor([precursor_mz], dtype=torch.float32)
        if mz_range > 0:
            precursor_tensor = (precursor_tensor - mz_min) / mz_range
    
    return peaks, precursor_tensor, mask


def pad_peak_batch(
    peak_lists: list[torch.Tensor],
    max_peaks: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a batch of variable-length peak lists to a fixed length.
    
    Args:
        peak_lists: List of (num_peaks, 2) tensors
        max_peaks: Maximum number of peaks; if None, uses max in batch
    
    Returns:
        padded_peaks: (batch, max_peaks, 2) tensor
        mask: (batch, max_peaks) boolean mask where True = valid, False = padded
    """
    if max_peaks is None:
        max_peaks = max(p.shape[0] for p in peak_lists) if peak_lists else 0
    
    batch_size = len(peak_lists)
    padded = torch.zeros(batch_size, max_peaks, 2, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_peaks, dtype=torch.bool)
    
    for i, peaks in enumerate(peak_lists):
        num_peaks = min(peaks.shape[0], max_peaks)
        padded[i, :num_peaks] = peaks[:num_peaks]
        mask[i, :num_peaks] = True
    
    return padded, mask
