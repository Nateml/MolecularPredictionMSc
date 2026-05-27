"""
Example: Using MultitaskCNNWithDreams in your training pipeline

This script demonstrates how to integrate the DreaMS encoder with your existing
training infrastructure. It's designed to be a drop-in replacement for your
existing run_training_cnn.py with minimal changes.
"""

from __future__ import annotations

import json
import torch
from pathlib import Path
from loguru import logger

from src.config import config
from src.model.model import MultitaskCNN, MultitaskCNNWithDreams
from src.data.datasets import dataset_from_schema, RawFeatureDataset, SpectrumDataset
from src.data.split import (
    split_dataset,
    splits_dir,
    splits_exist,
    load_splits,
    validate_splits_meta,
    TRAIN_FRACTION,
    VAL_FRACTION,
    CALIBRATION_FRACTION,
    TEST_FRACTION,
    SEED
)
from src.model.train import train
from src.model.conformal import calibrate_model
from torch.utils.data import DataLoader


def main_with_dreams_encoder(
    use_pretrained: bool = False,
    pretrained_path: str | None = None,
    freeze_encoder_epochs: int = 0,
):
    """
    Train MultitaskCNN with DreaMS encoder backbone.
    
    Args:
        use_pretrained: Whether to load pretrained encoder weights
        pretrained_path: Path to pretrained DreaMS encoder checkpoint
        freeze_encoder_epochs: Number of epochs to freeze encoder weights
                               (useful for transfer learning). Set to 0 to train normally.
    """
    data_path = config.data.resolved_dataset_path()
    schema_path = config.data.resolved_schema_path()
    output_path = config.checkpointing.resolved_save_dir() / "multitask_cnn_dreams.pt"

    logger.info(f"Building dataset from {data_path}...")
    dataset = dataset_from_schema(
        data_path=data_path,
        schema_path=schema_path,
        spectrum_len=config.model.spectrum_length,
    )
    logger.info(f"Dataset type: {type(dataset).__name__}, {len(dataset)} records")

    logger.info("Constructing DreaMS encoder model...")
    if isinstance(dataset, RawFeatureDataset):
        logger.info("Dataset is RawFeatureDataset, using QRegression heads.")
        model = MultitaskCNNWithDreams(
            spectrum_len=config.model.spectrum_length,
            mz_min=0.0,
            mz_max=1000.0,
            num_binary=dataset.maccs_length,
            categorical_sizes=dataset.categorical_sizes(),
            num_regression=dataset.n_continuous,
            num_quantiles=config.model.num_quantiles,
            dropout=config.model.dropout,
            # DreaMS encoder config
            hidden_dim=256,
            num_transformer_layers=4,
            num_attention_heads=8,
            ff_dim=1024,
            use_precursor_token=True,
            # Load pretrained if available
            pretrained_encoder_path=pretrained_path if use_pretrained else None,
        )
    else:
        logger.info("Dataset is SpectrumDataset, using standard regression heads.")
        model = MultitaskCNNWithDreams(
            spectrum_len=config.model.spectrum_length,
            mz_min=0.0,
            mz_max=1000.0,
            num_binary=dataset.maccs_length,
            categorical_sizes=dataset.categorical_sizes(),
            num_regression=0,
            dropout=config.model.dropout,
            # DreaMS encoder config
            hidden_dim=256,
            num_transformer_layers=4,
            num_attention_heads=8,
            ff_dim=1024,
            use_precursor_token=True,
            pretrained_encoder_path=pretrained_path if use_pretrained else None,
        )

    logger.info(f"Model:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup train/val/test splits
    if splits_exist(config.data.splits_dir()):
        logger.info("Loading existing splits...")
        splits = load_splits(config.data.splits_dir())
        validate_splits_meta(splits, len(dataset), SEED)
    else:
        logger.info("Creating new splits...")
        splits = split_dataset(
            dataset,
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            calibration_fraction=CALIBRATION_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=SEED,
        )
        splits_dir().mkdir(parents=True, exist_ok=True)
        splits_path = splits_dir() / "meta.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f)
        logger.info(f"Splits saved to {splits_path}")

    # Create dataloaders
    train_loader = DataLoader(
        dataset.subset(splits["train"]),
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
    )
    val_loader = DataLoader(
        dataset.subset(splits["val"]),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    test_loader = DataLoader(
        dataset.subset(splits["test"]),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    calibration_loader = DataLoader(
        dataset.subset(splits["calibration"]),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )

    # Training with potential encoder freezing (transfer learning)
    if freeze_encoder_epochs > 0:
        logger.info(f"Freezing encoder for first {freeze_encoder_epochs} epochs (transfer learning)")
        model.freeze_encoder()
        initial_epochs = freeze_encoder_epochs
    else:
        initial_epochs = 0

    # Train model
    logger.info(f"Starting training with {config.training.num_epochs} epochs...")
    model = train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.num_epochs,
        learning_rate=config.training.learning_rate,
        save_path=output_path,
        device=config.training.device,
    )

    # Unfreeze encoder if it was frozen
    if freeze_encoder_epochs > 0:
        logger.info("Unfreezing encoder for fine-tuning")
        model.unfreeze_encoder()
        # Could continue training with lower learning rate here if desired
        # model = train(..., learning_rate=config.training.learning_rate * 0.1, ...)

    # Test model
    logger.info("Evaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, config.training.device)
    logger.info(f"Test metrics: {test_metrics}")

    # Calibrate (conformal prediction)
    logger.info("Calibrating model with conformal prediction...")
    conformal_thresholds = calibrate_model(
        model,
        calibration_loader,
        device=config.training.device,
        num_quantiles=config.model.num_quantiles,
    )
    
    # Save calibration thresholds
    calibration_path = config.checkpointing.resolved_save_dir() / "conformal_thresholds.pt"
    torch.save(conformal_thresholds, calibration_path)
    logger.info(f"Calibration thresholds saved to {calibration_path}")

    logger.info(f"Model saved to {output_path}")
    logger.info("Training complete!")

    return model


def evaluate_model(model, dataloader, device):
    """Simple evaluation function for testing."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            x, targets = batch
            x = x.to(device)
            
            # Move targets to device
            targets_device = {}
            for key, val in targets.items():
                if isinstance(val, torch.Tensor):
                    targets_device[key] = val.to(device)
                elif isinstance(val, list):
                    targets_device[key] = [v.to(device) for v in val]
                else:
                    targets_device[key] = val

            # Forward pass
            outputs = model(x)
            # Compute loss (you should use your actual loss function)
            # For now, just a placeholder
            num_batches += 1

    return {"avg_loss": total_loss / max(num_batches, 1)}


# ============================================================================
# Configuration presets for different use cases
# ============================================================================

class DreaMSConfigs:
    """Preset configurations for DreaMS encoder."""
    
    @staticmethod
    def small():
        """Small model for quick experimentation."""
        return {
            "hidden_dim": 128,
            "num_transformer_layers": 2,
            "num_attention_heads": 4,
            "ff_dim": 512,
        }
    
    @staticmethod
    def medium():
        """Balanced model - recommended for most use cases."""
        return {
            "hidden_dim": 256,
            "num_transformer_layers": 4,
            "num_attention_heads": 8,
            "ff_dim": 1024,
        }
    
    @staticmethod
    def large():
        """Large model for best performance (requires more memory)."""
        return {
            "hidden_dim": 512,
            "num_transformer_layers": 6,
            "num_attention_heads": 8,
            "ff_dim": 2048,
        }


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # OPTION 1: Train from scratch with DreaMS encoder
    logger.info("=" * 80)
    logger.info("Training MultitaskCNN with DreaMS encoder (from scratch)")
    logger.info("=" * 80)
    model = main_with_dreams_encoder(
        use_pretrained=False,
        pretrained_path=None,
        freeze_encoder_epochs=0,  # Train all layers
    )

    # OPTION 2: Transfer learning - freeze encoder for first N epochs
    # logger.info("=" * 80)
    # logger.info("Transfer learning: Training heads while encoder is frozen")
    # logger.info("=" * 80)
    # model = main_with_dreams_encoder(
    #     use_pretrained=True,
    #     pretrained_path="path/to/pretrained/encoder.pt",
    #     freeze_encoder_epochs=5,  # Freeze for 5 epochs, then fine-tune
    # )

    # OPTION 3: Fine-tune with pretrained weights
    # logger.info("=" * 80)
    # logger.info("Fine-tuning: Using pretrained encoder")
    # logger.info("=" * 80)
    # model = main_with_dreams_encoder(
    #     use_pretrained=True,
    #     pretrained_path="path/to/pretrained/encoder.pt",
    #     freeze_encoder_epochs=0,  # Train everything from start
    # )
