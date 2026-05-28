# DreaMS Pretrained Weights Guide

## Overview

This guide explains how to obtain, use, and manage pretrained DreaMS encoder weights for your multitask model.

## Current Status (May 2026)

As of this date, there are **several sources** for DreaMS-like models:

### 1. **Official DreaMS Repository** (Most Reliable)
- **Repository**: Look for the official DreaMS paper's GitHub repository
- **Citation**: The DreaMS paper (if published) will contain official implementation
- **Weights**: May be available in their `pretrained_models/` directory

### 2. **Hugging Face Model Hub**
- **Status**: Not yet available as of your implementation date
- **Check**: `huggingface.co/NateML/` or similar researcher's organization
- **Usage**: Once available, use:
  ```python
  from src.model.dreams_encoder import load_pretrained_dreams
  encoder = load_pretrained_dreams("NateML/DreaMS-pretrained")
  ```

### 3. **Your Own Pretrained Weights**
- **Recommended for most users**
- **Process**: Train on your own dataset (or large MS dataset), save weights
- **Advantage**: Optimized for your specific domain

## Option 1: Using Weights from Official DreaMS Repository

### Step 1: Find the Official Repository
```bash
# DreaMS likely available at:
# https://github.com/[author]/DreaMS
# or through the paper supplementary materials
```

### Step 2: Download Pretrained Weights
```bash
# Clone the repo
git clone https://github.com/[author]/DreaMS.git
cd DreaMS

# Look for pretrained weights
ls -la models/pretrained/
# or
ls -la checkpoints/
```

### Step 3: Convert to Your Format (if needed)
```python
import torch
from src.model.dreams_encoder import DreaMSEncoder

# Load original DreaMS weights
original_checkpoint = torch.load("DreaMS/checkpoints/dreams_base.pt")

# If the architecture differs, manually adapt:
encoder = DreaMSEncoder(hidden_dim=256, num_transformer_layers=4)

# If compatible, load directly
encoder.load_state_dict(original_checkpoint)

# Save in your format
torch.save({"state_dict": encoder.state_dict()}, 
           "checkpoints/dreams_base_adapted.pt")
```

## Option 2: Loading from Hugging Face Hub

### When Official Weights Become Available
```python
from src.model.dreams_weights import get_pretrained_encoder

# Once on Hugging Face:
encoder = get_pretrained_encoder("dreams-base", device="cuda")

# Or directly:
from transformers import AutoModel
encoder = AutoModel.from_pretrained("NateML/DreaMS-base")
```

## Option 3: Training Your Own Pretrained Weights

### Recommended Approach
Since pretrained weights may not be immediately available, **training your own is practical**:

```python
# 1. Train on large MS dataset (MoNA, NIST, etc.)
python src/model/run_training_dreams.py --dataset mona_large.json

# 2. Save just the encoder
from src.model.dreams_weights import save_trained_encoder
save_trained_encoder(model.encoder, "checkpoints/my_pretrained_encoder.pt")

# 3. Use in downstream tasks
from src.model.dreams_weights import load_saved_encoder
encoder = load_saved_encoder("checkpoints/my_pretrained_encoder.pt")

# 4. Fine-tune on your specific task
model.encoder = encoder
model.freeze_encoder()  # Freeze for first epoch
# ... train heads ...
model.unfreeze_encoder()
# ... fine-tune everything ...
```

### Script to Train Pretrained Encoder

Create `src/model/train_pretrained_encoder.py`:

```python
"""
Train a DreaMS encoder on a large MS dataset for pretraining.
"""

import torch
from pathlib import Path
from loguru import logger
from src.model.model import MultitaskCNNWithDreams
from src.data.datasets import dataset_from_schema
from src.model.dreams_weights import save_trained_encoder
from torch.utils.data import DataLoader

def train_encoder_only(
    data_path: str,
    schema_path: str,
    output_path: str = "checkpoints/dreams_pretrained.pt",
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
):
    """Train DreaMS encoder on large MS dataset."""
    
    # Load dataset
    dataset = dataset_from_schema(
        data_path=data_path,
        schema_path=schema_path,
        spectrum_len=1000,
    )
    
    # Create model
    model = MultitaskCNNWithDreams(
        spectrum_len=1000,
        num_binary=dataset.maccs_length,
        categorical_sizes=dataset.categorical_sizes(),
        hidden_dim=256,
        num_transformer_layers=4,
    )
    
    # Training loop (use your existing train function)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        for batch_idx, (x, targets) in enumerate(dataloader):
            x = x.to(device)
            outputs = model(x)
            
            # Compute loss (your loss function here)
            # loss = ...
            
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            if batch_idx % 100 == 0:
                logger.info(f"Epoch {epoch}, Batch {batch_idx}")
    
    # Save pretrained encoder
    save_trained_encoder(
        model.encoder,
        save_path=output_path,
        training_info={
            "epochs": num_epochs,
            "dataset": data_path,
            "learning_rate": learning_rate,
        }
    )
    
    logger.info(f"Pretrained encoder saved to {output_path}")
```

## Option 4: Manual Weight Management

### Save Individual Encoder Weights
```python
# After training
encoder_state = model.encoder.state_dict()
torch.save(encoder_state, "checkpoints/encoder_weights_only.pt")

# Load as part of new model
new_model = MultitaskCNNWithDreams(...)
new_model.encoder.load_state_dict(torch.load("checkpoints/encoder_weights_only.pt"))
```

### Save Full Checkpoint with Metadata
```python
from src.model.dreams_weights import DreaMSWeightsManager

manager = DreaMSWeightsManager("checkpoints")
manager.save_checkpoint(
    model.encoder,
    save_path="checkpoints/full_checkpoint.pt",
    metadata={
        "model_name": "DreaMSEncoder",
        "hidden_dim": 256,
        "num_layers": 4,
        "trained_on": "MoNA",
        "epochs": 100,
        "final_loss": 0.023,
    }
)
```

## Using Pretrained Weights in Your Workflow

### During Initialization
```python
model = MultitaskCNNWithDreams(
    ...,
    pretrained_encoder_path="checkpoints/dreams_encoder.pt"
)
```

### After Initialization
```python
model = MultitaskCNNWithDreams(...)

# Load checkpoint
checkpoint = torch.load("checkpoints/dreams_encoder.pt")
model.encoder.load_state_dict(checkpoint["state_dict"])

# Or use helper function
from src.model.dreams_weights import load_saved_encoder
model.encoder = load_saved_encoder("checkpoints/dreams_encoder.pt")
```

## Transfer Learning Strategies

### Strategy 1: Head-Only Training
```python
# Freeze encoder, train only new heads for your specific task
model.freeze_encoder()

for epoch in range(5):
    # Train with frozen encoder
    train_one_epoch(model, train_loader)

print(f"Trainable params: {count_trainable_params(model):,}")
# Output: Only head parameters are trainable
```

### Strategy 2: Progressive Unfreezing
```python
# Start with frozen encoder, gradually unfreeze
model.freeze_encoder()

for stage in range(3):
    logger.info(f"Training stage {stage + 1}...")
    
    if stage > 0:
        model.unfreeze_encoder()  # Gradually unfreeze
    
    train_epochs(model, train_loader, num_epochs=5)
```

### Strategy 3: Discriminative Learning Rates
```python
from src.model.dreams_weights import get_discriminative_learning_rates

# Lower learning rate for pretrained encoder, higher for new heads
param_groups = get_discriminative_learning_rates(
    model,
    base_lr=1e-4,
    encoder_lr_factor=0.1  # 10x lower for encoder
)

optimizer = torch.optim.Adam(param_groups)
```

## Troubleshooting

### Problem: Weight Incompatibility
```python
# Solution: Check dimensions match
encoder_weights = torch.load("encoder.pt")
print(encoder_weights.keys())

# If using different architecture, create adapter layer
class EncoderAdapter(nn.Module):
    def __init__(self, old_dim, new_dim):
        super().__init__()
        self.proj = nn.Linear(old_dim, new_dim)
    
    def forward(self, x):
        return self.proj(x)
```

### Problem: CUDA Memory Issues with Pretrained
```python
# Solution: Reduce batch size or model size
batch_size = 32  # Reduce from 64
# Or use gradient accumulation
num_accumulation_steps = 2
```

### Problem: Slow Convergence After Loading Pretrained
```python
# Solution: Use proper learning rate warmup
from torch.optim.lr_scheduler import LinearLR

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)

for epoch in range(num_epochs):
    train_one_epoch(model, train_loader)
    warmup_scheduler.step()
```

## Comparison: Pretrained vs From-Scratch

| Aspect | Pretrained | From-Scratch |
|--------|-----------|------------|
| Initial Performance | High | Low |
| Convergence Speed | Fast | Slow |
| Data Requirements | Lower | Higher |
| Fine-tuning Time | 5-10 epochs | 50-100 epochs |
| Memory Usage | Higher (full model) | Lower |
| Customization | Limited | Full |

## Recommended Workflow

### For Optimal Results:
1. **Start with pretrained weights** if available
2. **Freeze encoder**, train heads for 2-3 epochs
3. **Unfreeze and fine-tune** with lower learning rate
4. **Monitor validation** to prevent overfitting
5. **Save final model** for inference

### Example Complete Script:
```python
from src.model.dreams_weights import load_saved_encoder
from src.model.model import MultitaskCNNWithDreams

# Create model with pretrained encoder
model = MultitaskCNNWithDreams(...)
pretrained_encoder = load_saved_encoder("pretrained/dreams_base.pt")
model.encoder = pretrained_encoder

# Stage 1: Train heads only
model.freeze_encoder()
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
for epoch in range(3):
    train_one_epoch(model, train_loader, optimizer)

# Stage 2: Fine-tune everything
model.unfreeze_encoder()
optimizer = Adam(model.parameters(), lr=1e-5)
for epoch in range(20):
    train_one_epoch(model, train_loader, optimizer)
    validate(model, val_loader)

# Save final model
torch.save(model.state_dict(), "final_model.pt")
```

## Where to Get Help

1. **Check official DreaMS repository** for latest weight availability
2. **Hugging Face Hub**: Look for `NateML/*` or author's organization
3. **Paper supplementary materials**: Often includes download links
4. **Your own training**: Most reliable option for custom models

## Future Updates

- Watch for official Hugging Face release of DreaMS weights
- Community-trained models may become available
- Consider creating your own pretrained weights library

## Questions?

If official pretrained weights aren't available:
1. Train your own encoder on MoNA or NIST dataset
2. Use the `train_encoder_only()` script provided
3. Save with `save_trained_encoder()` for reuse
4. Share community-trained weights when ready
