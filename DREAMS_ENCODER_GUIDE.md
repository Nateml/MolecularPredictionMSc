# DreaMS Encoder Integration Guide

## Overview

You now have a `MultitaskCNNWithDreams` model that replaces your CNN backbone with a peak-list transformer encoder (inspired by DreaMS architecture). This encoder:

- **Accepts variable-length** (m/z, intensity) peak lists
- **Uses self-attention** over peaks for better feature learning
- **Preserves your existing multitask heads** (binary, categorical, quantile regression)
- **Supports pretrained weights** from DreaMS or trained models

## Key Features

### 1. **Flexible Input Handling**
The model can accept inputs in multiple formats:
- Dense spectrum vectors (your current format) → automatically converted to peak lists
- Raw peak list tensors (batch, max_peaks, 2)
- Optional precursor m/z values
- Variable-length sequences with attention masks

### 2. **Peak List Transformer Architecture**
```
Input: (m/z, intensity) pairs
  ↓
Peak Tokenizer: 2D → hidden_dim embedding
  ↓
Positional Encoding: Add sequence position information
  ↓
Transformer Encoder: Multi-head self-attention over peaks
  ↓
Global Average Pooling: Variable-length → fixed-size
  ↓
Multitask Heads: Binary, categorical, quantile regression
```

### 3. **Pretrained Weights Support**
Can load pretrained DreaMS encoder weights with:
```python
model = MultitaskCNNWithDreams(
    ...,
    pretrained_encoder_path="path/to/pretrained/encoder.pt"
)
```

## Usage Examples

### Basic Usage (with Dense Spectrum Input)

```python
from src.model.model import MultitaskCNNWithDreams, categorical_sizes_from_df
from src.data.datasets import ContinuousDataset

# Prepare categorical sizes from your data
feature_df = ...  # your feature dataframe
binary_cols = [...]
categorical_sizes = categorical_sizes_from_df(feature_df, binary_cols)

# Create model with DreaMS encoder
model = MultitaskCNNWithDreams(
    spectrum_len=1000,
    mz_min=0.0,
    mz_max=1000.0,
    num_binary=166,
    categorical_sizes=categorical_sizes,
    num_regression=4,
    num_quantiles=3,
    hidden_dim=256,
    num_transformer_layers=4,
    num_attention_heads=8,
    dropout=0.2,
)

# Forward pass with dense spectrum (same as before)
dense_spectrum = torch.randn(batch_size, 1000)
output = model(dense_spectrum)

print(output["binary"].shape)        # (batch, 166)
print(output["categorical"][0].shape) # (batch, num_classes)
print(output["quantile"][0].shape)   # (batch, 3)
```

### With Precursor m/z Values

```python
# If your data includes precursor m/z
precursor_mz = torch.randn(batch_size)  # (batch,)

output = model(
    dense_spectrum,
    precursor_mz=precursor_mz
)
```

### With Variable-Length Peak Lists

```python
from src.model.dreams_encoder import spectrum_string_to_peaks, pad_peak_batch

# Convert your spectrum strings to peak lists
spectrum_strings = [...]  # from your dataset
peak_lists = []
for spec_str in spectrum_strings:
    peaks, precursor, _ = spectrum_string_to_peaks(
        spec_str, 
        mz_min=0.0, 
        mz_max=1000.0
    )
    peak_lists.append(peaks)

# Pad to fixed length
padded_peaks, attention_mask = pad_peak_batch(peak_lists, max_peaks=256)

# Forward pass with peak lists
output = model(
    x=padded_peaks,  # Already in (batch, max_peaks, 2) format
    attention_mask=attention_mask
)
```

### Loading Pretrained Weights

```python
# Option 1: Load during initialization
model = MultitaskCNNWithDreams(
    ...,
    pretrained_encoder_path="checkpoints/dreams_encoder.pt"
)

# Option 2: Load after initialization
model = MultitaskCNNWithDreams(...)
checkpoint = torch.load("checkpoints/dreams_encoder.pt")
model.encoder.load_state_dict(checkpoint)

# Option 3: From Hugging Face (when available)
from src.model.dreams_encoder import load_pretrained_dreams
pretrained = load_pretrained_dreams("NateML/DreaMS-pretrained")
model.encoder = pretrained
```

### Transfer Learning Workflows

```python
# Freeze encoder, train heads only
model.freeze_encoder()
# ... train for a few epochs ...

# Fine-tune everything
model.unfreeze_encoder()
learning_rate = 1e-5  # Lower LR for fine-tuning
# ... continue training ...
```

## Architecture Details

### DreaMSEncoder Parameters

- **hidden_dim** (256): Embedding dimension for peaks
- **num_transformer_layers** (4): Number of transformer blocks
- **num_attention_heads** (8): Heads in multi-head attention
- **ff_dim** (1024): Feed-forward network dimension
- **dropout** (0.2): Dropout rate
- **use_precursor_token** (True): Add special token for precursor m/z

### Recommended Configurations

#### Small Model (Fast Training)
```python
hidden_dim=128, num_transformer_layers=2, num_attention_heads=4, ff_dim=512
```

#### Medium Model (Balanced)
```python
hidden_dim=256, num_transformer_layers=4, num_attention_heads=8, ff_dim=1024
```

#### Large Model (Best Performance)
```python
hidden_dim=512, num_transformer_layers=6, num_attention_heads=8, ff_dim=2048
```

## Key Differences from CNN

| Aspect | CNN | DreaMS Transformer |
|--------|-----|-------------------|
| Input | Fixed-length dense vector | Variable-length peak list |
| Information Loss | Binning m/z into fixed size | Preserves exact m/z values |
| Adaptation | Fixed receptive field | Global attention over peaks |
| Pretraining | None readily available | Potential for pretrained weights |
| Inference Speed | Slightly faster | Slightly slower (variable length) |
| Memory | Lower | Higher (quadratic in seq_len) |

## Handling Variable-Length Sequences

The model handles variable-length peak lists efficiently:

1. **Attention Masking**: Padding positions are masked in transformer attention
2. **Adaptive Pooling**: Only valid peaks contribute to global pooling
3. **Efficient Implementation**: Uses PyTorch transformer with nested tensor support

```python
# Different spectra may have different numbers of peaks
spectrum_1 = "10.5:100 20.3:50 30.1:75"   # 3 peaks
spectrum_2 = "15.2:120"                    # 1 peak
spectrum_3 = "5.1:10 10.2:20 15.3:30 20.4:40"  # 4 peaks

# All can be batched together with attention masks
# The model will handle variable lengths automatically
```

## Integration with Training Scripts

### Minimal Training Loop Update

If you're using your existing `run_training.py`, minimal changes needed:

```python
# OLD:
model = MultitaskCNN(...)

# NEW:
model = MultitaskCNNWithDreams(
    ...,
    hidden_dim=256,
    num_transformer_layers=4,
    pretrained_encoder_path=None  # or path to pretrained weights
)

# Rest of training loop remains the same!
# The model accepts the same input format as MultitaskCNN
```

## Performance Tips

1. **Batch Size**: Variable-length sequences work best with moderate batch sizes (16-64)
2. **Max Peaks**: Limit max peaks if you have outliers with many peaks
3. **Learning Rate**: Start with 1e-4 for training from scratch, 1e-5 for fine-tuning
4. **Warmup**: Use learning rate warmup for transformer stability

```python
# Example with warmup
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

for epoch in range(num_epochs):
    # training loop
    scheduler.step()
```

## Saving and Loading Models

```python
# Save checkpoint
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
}
torch.save(checkpoint, 'checkpoint.pt')

# Load checkpoint
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

## Troubleshooting

### Out of Memory with Large Peak Lists
```python
# Reduce max_peaks or hidden_dim
padded_peaks, mask = pad_peak_batch(peak_lists, max_peaks=128)  # Instead of 256
```

### Variable-Length Sequences Causing Issues
```python
# Ensure attention_mask is properly computed and passed
attention_mask = ~torch.isnan(peaks[:,:,0])  # True where valid
output = model(peaks, attention_mask=attention_mask)
```

### Pretrained Weights Not Loading
```python
# Check the checkpoint format
checkpoint = torch.load("path/to/weights.pt")
print(checkpoint.keys())  # Should contain 'state_dict' or model weights

# Load manually if needed
encoder_state = checkpoint.get('encoder', checkpoint.get('state_dict'))
model.encoder.load_state_dict(encoder_state)
```

## Next Steps

1. **Replace your training script** to use `MultitaskCNNWithDreams`
2. **Compare performance** with baseline CNN model
3. **Fine-tune hyperparameters** based on your dataset
4. **Explore pretrained weights** once available from official DreaMS sources
5. **Consider data augmentation** for peak lists (peak dropping, intensity perturbation)

## References

- Original paper: DreaMS architecture based on transformer encoders for mass spectrometry
- PyTorch Transformer: https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- Your existing heads: Unchanged from original implementation

## Questions or Issues?

- Check that `categorical_sizes` are computed correctly
- Verify spectrum preprocessing (m/z range, intensity normalization)
- Ensure batch handling for variable-length sequences matches your dataloader
