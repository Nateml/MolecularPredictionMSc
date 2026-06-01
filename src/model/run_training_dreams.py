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
    fractions = {
        "train": TRAIN_FRACTION,
        "val": VAL_FRACTION,
        "calibration": CALIBRATION_FRACTION,
        "test": TEST_FRACTION,
    }
    split_dir = splits_dir(data_path, schema_path, fractions, SEED)
    
    if splits_exist(split_dir):
        logger.info(f"Loading existing splits from {split_dir}...")
        train_ds, val_ds, calibration_ds, test_ds = load_splits(split_dir)
    else:
        logger.info(f"No splits found at {split_dir}, creating new splits...")
        train_ds, val_ds, calibration_ds, test_ds = split_dataset(
            dataset,
            train_fraction=TRAIN_FRACTION,
            val_fraction=VAL_FRACTION,
            calibration_fraction=CALIBRATION_FRACTION,
            test_fraction=TEST_FRACTION,
            seed=SEED,
            check_label_consistency=True,
        )
        split_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, "
        f"Calibration samples: {len(calibration_ds)}, Test samples: {len(test_ds)}"
    )

    # Normalize continuous targets if applicable
    norm_stats = None
    if isinstance(dataset, RawFeatureDataset) and hasattr(train_ds, "normalise_continuous"):
        logger.info("Normalising regression targets...")
        mean, std = train_ds.normalise_continuous()
        _, _ = val_ds.normalise_continuous(mean, std)
        _, _ = calibration_ds.normalise_continuous(mean, std)
        _, _ = test_ds.normalise_continuous(mean, std)
        norm_stats = {"mean": mean.tolist() if hasattr(mean, "tolist") else mean,
                      "std": std.tolist() if hasattr(std, "tolist") else std}

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Train model (with potential encoder freezing for transfer learning)
    if freeze_encoder_epochs > 0:
        logger.info(f"Training with encoder freezing: {freeze_encoder_epochs} epochs frozen, then fine-tuning")
    
    logger.info(f"Starting training with {config.training.num_epochs} epochs...")
    model = train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        n_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        lr=config.training.learning_rate,
        device=device,
        output_path=output_path,
        continuous_stats=norm_stats,
    )

    # Compute conformal thresholds from calibration set
    logger.info("Computing conformal thresholds from calibration set...")
    calib_loader = DataLoader(calibration_ds, batch_size=config.training.batch_size, shuffle=False)
    thresholds = calibrate_model(model, calib_loader, alpha=config.conformal.alpha)
    thresholds_path = output_path.parent / "conformal_thresholds.pt"
    torch.save(thresholds, thresholds_path)
    logger.info(f"Conformal thresholds saved to {thresholds_path}")

    logger.info(f"Model saved to {output_path}")
    logger.info("Training complete!")

    return model


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
