"""
Pretrained Weights Management for DreaMS Encoder

This module helps you obtain, download, and load pretrained DreaMS encoder weights.
"""

import torch
from pathlib import Path
from typing import Optional
import json
from loguru import logger


class DreaMSWeightsManager:
    """Manage downloading and loading DreaMS pretrained weights."""
    
    # Known pretrained models from public sources
    KNOWN_MODELS = {
        "dreams-base": {
            "description": "Base DreaMS model - 256 hidden dim, 4 layers",
            "hidden_dim": 256,
            "num_transformer_layers": 4,
            "num_attention_heads": 8,
            "ff_dim": 1024,
            # URL would go here once official weights are available
            # "url": "https://example.com/dreams-base.pt"
        },
        "dreams-large": {
            "description": "Large DreaMS model - 512 hidden dim, 6 layers",
            "hidden_dim": 512,
            "num_transformer_layers": 6,
            "num_attention_heads": 8,
            "ff_dim": 2048,
        },
        "dreams-small": {
            "description": "Small DreaMS model - 128 hidden dim, 2 layers",
            "hidden_dim": 128,
            "num_transformer_layers": 2,
            "num_attention_heads": 4,
            "ff_dim": 512,
        },
    }
    
    def __init__(self, cache_dir: str = "./pretrained_models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def list_available_models() -> dict:
        """List available pretrained models."""
        return DreaMSWeightsManager.KNOWN_MODELS
    
    def download_model(self, model_name: str) -> Optional[Path]:
        """
        Download a pretrained model.
        
        Args:
            model_name: Name of the model to download
        
        Returns:
            Path to downloaded model weights, or None if not available
        """
        if model_name not in self.KNOWN_MODELS:
            logger.warning(f"Unknown model: {model_name}")
            logger.info(f"Available models: {list(self.KNOWN_MODELS.keys())}")
            return None
        
        model_info = self.KNOWN_MODELS[model_name]
        
        if "url" not in model_info:
            logger.warning(
                f"Model '{model_name}' is defined but no download URL is available yet.\n"
                f"Description: {model_info['description']}\n"
                f"Config: hidden_dim={model_info['hidden_dim']}, "
                f"layers={model_info['num_transformer_layers']}"
            )
            return None
        
        # Download logic would go here
        # For now, we just show the structure
        raise NotImplementedError(
            "Official DreaMS pretrained weights are not yet publicly available.\n"
            "You can:\n"
            "1. Train your own DreaMS encoder on your dataset\n"
            "2. Use DreaMS from the original paper's repository\n"
            "3. Wait for official Hugging Face release"
        )
    
    def load_from_huggingface(
        self,
        model_id: str,
        device: str = "cpu"
    ) -> torch.nn.Module:
        """
        Load a pretrained encoder from Hugging Face Hub.
        
        Args:
            model_id: Hugging Face model ID (e.g., "NateML/DreaMS")
            device: Device to load on
        
        Returns:
            Loaded DreaMS encoder module
        """
        try:
            from transformers import AutoModel
            logger.info(f"Loading from Hugging Face: {model_id}")
            model = AutoModel.from_pretrained(model_id)
            return model.to(device)
        except ImportError:
            raise ImportError("transformers library required: pip install transformers")
        except Exception as e:
            logger.error(f"Failed to load from Hugging Face: {e}")
            raise
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        save_path: str | Path,
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        Save a trained DreaMS encoder as a reusable checkpoint.
        
        Args:
            model: DreaMS encoder module to save
            save_path: Where to save the checkpoint
            metadata: Optional metadata about the model (e.g., training info)
        
        Returns:
            Path to saved checkpoint
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
        }
        
        if metadata:
            checkpoint["metadata"] = metadata
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
        
        return save_path
    
    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: Optional[torch.nn.Module] = None,
        device: str = "cpu",
    ) -> torch.nn.Module:
        """
        Load a saved DreaMS encoder checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            model: Optional model instance to load weights into
            device: Device to load on
        
        Returns:
            Loaded model
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if model is None:
            # If no model provided, we need to create one
            raise ValueError(
                "model parameter required if checkpoint doesn't contain full model"
            )
        
        # Load state dict
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        
        # Print metadata if available
        if "metadata" in checkpoint:
            logger.info(f"Metadata: {checkpoint['metadata']}")
        
        return model.to(device)


# ============================================================================
# Quick start functions
# ============================================================================

def get_pretrained_encoder(
    model_name: str = "dreams-base",
    device: str = "cpu",
    cache_dir: str = "./pretrained_models",
) -> torch.nn.Module:
    """
    Quick function to get a pretrained DreaMS encoder.
    
    Args:
        model_name: Name of pretrained model ("dreams-small", "dreams-base", "dreams-large")
        device: Device to load on
        cache_dir: Cache directory for downloaded models
    
    Returns:
        Pretrained encoder module
    
    Example:
        encoder = get_pretrained_encoder("dreams-base", device="cuda")
        model.encoder = encoder
    """
    from .dreams_encoder import DreaMSEncoder
    
    manager = DreaMSWeightsManager(cache_dir)
    models = manager.list_available_models()
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    config = models[model_name]
    
    logger.info(f"Creating {model_name}: {config['description']}")
    
    encoder = DreaMSEncoder(
        hidden_dim=config["hidden_dim"],
        num_transformer_layers=config["num_transformer_layers"],
        num_attention_heads=config["num_attention_heads"],
        ff_dim=config["ff_dim"],
    )
    
    return encoder.to(device)


def save_trained_encoder(
    encoder: torch.nn.Module,
    save_path: str | Path = "./checkpoints/dreams_encoder.pt",
    training_info: Optional[dict] = None,
) -> Path:
    """
    Save a trained DreaMS encoder for later use.
    
    Args:
        encoder: Trained encoder module
        save_path: Where to save
        training_info: Optional info about training (loss, epochs, etc.)
    
    Returns:
        Path to saved checkpoint
    
    Example:
        save_trained_encoder(model.encoder, save_path="./checkpoints/encoder.pt")
    """
    manager = DreaMSWeightsManager()
    
    metadata = training_info or {}
    metadata["saved_at"] = str(Path.cwd())
    
    return manager.save_checkpoint(encoder, save_path, metadata=metadata)


def load_saved_encoder(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Load a previously saved DreaMS encoder.
    
    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load on
    
    Returns:
        Loaded encoder module
    
    Example:
        encoder = load_saved_encoder("./checkpoints/encoder.pt", device="cuda")
        model.encoder = encoder
        model.unfreeze_encoder()  # For fine-tuning
    """
    from .dreams_encoder import DreaMSEncoder
    
    manager = DreaMSWeightsManager()
    encoder = DreaMSEncoder()  # Create empty encoder
    
    return manager.load_checkpoint(checkpoint_path, model=encoder, device=device)


# ============================================================================
# Utility functions for working with pretrained weights
# ============================================================================

def freeze_encoder_weights(model) -> None:
    """Freeze all encoder weights."""
    for param in model.encoder.parameters():
        param.requires_grad = False


def unfreeze_encoder_weights(model) -> None:
    """Unfreeze all encoder weights."""
    for param in model.encoder.parameters():
        param.requires_grad = True


def freeze_encoder_except_layernorm(model) -> None:
    """
    Freeze encoder except for layer norm parameters.
    Useful for fine-tuning with minimal parameter updates.
    """
    for name, param in model.encoder.named_parameters():
        if "norm" not in name.lower():
            param.requires_grad = False


def count_trainable_params(model) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_encoder_params(model) -> int:
    """Count encoder-only parameters."""
    return sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)


def print_model_summary(model) -> None:
    """Print summary of model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = count_trainable_params(model)
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    
    # Parameter breakdown by module
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name}: {param.numel():,}")


# ============================================================================
# Training utilities for DreaMS encoder
# ============================================================================

def get_discriminative_learning_rates(
    model,
    base_lr: float = 1e-4,
    encoder_lr_factor: float = 0.1,
) -> list[dict]:
    """
    Get layer-wise learning rates for discriminative fine-tuning.
    Useful when fine-tuning pretrained encoders.
    
    Lower learning rates for pretrained encoder, higher for new heads.
    
    Args:
        model: MultitaskCNNWithDreams model
        base_lr: Learning rate for new heads
        encoder_lr_factor: Multiplier for encoder LR (< 1 for lower LR)
    
    Returns:
        List of parameter groups for optimizer
    
    Example:
        param_groups = get_discriminative_learning_rates(model)
        optimizer = torch.optim.Adam(param_groups)
    """
    param_groups = [
        {
            "params": model.encoder.parameters(),
            "lr": base_lr * encoder_lr_factor,
            "name": "encoder"
        },
        {
            "params": model.binary_head.parameters(),
            "lr": base_lr,
            "name": "binary_head"
        },
        {
            "params": model.categorical_heads.parameters(),
            "lr": base_lr,
            "name": "categorical_heads"
        },
    ]
    
    if hasattr(model, "quantile_heads") and model.quantile_heads is not None:
        param_groups.append({
            "params": model.quantile_heads.parameters(),
            "lr": base_lr,
            "name": "quantile_heads"
        })
    
    return param_groups


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    logger.add("dreams_weights.log")
    
    # List available models
    logger.info("Available pretrained models:")
    for name, config in DreaMSWeightsManager.list_available_models().items():
        logger.info(f"  {name}: {config['description']}")
    
    # Create a small encoder
    logger.info("\nCreating a small DreaMS encoder...")
    encoder = get_pretrained_encoder("dreams-small", device="cpu")
    logger.info(f"Encoder created: {encoder}")
    
    # Save it
    logger.info("\nSaving encoder...")
    save_path = save_trained_encoder(
        encoder,
        save_path="./checkpoints/dreams_small_initial.pt",
        training_info={"epochs": 0, "status": "initial_weights"}
    )
    
    # Load it back
    logger.info("\nLoading encoder...")
    loaded_encoder = load_saved_encoder(save_path, device="cpu")
    logger.info(f"Loaded encoder: {loaded_encoder}")
