from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
import torch.nn as nn

import sys
from pathlib import Path as _Path
import torch

# If this file is executed directly (``python neural_net/run_training.py``),
# Python sets ``sys.path[0]`` to the ``neural_net`` directory which breaks
# package-relative imports like ``from neural_net.model import ...``.
# When no package is set, prepend the project root so imports resolve.
if __package__ is None:
    sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from neural_net.model import MultitaskMLP
from neural_net.onehot_decoded_dataset import OneHotDecodedDataset
from neural_net.mixed_dataset import MixedFeatureDataset, MixedRawFeatureDataset
from neural_net.train import split_dataset, evaluate, compute_loss, train
from torch.utils.data import DataLoader
import torch.optim as optim


# --- Hard-coded configuration (intentionally not generalized) ---
#DATA_PATH = Path("Data Processing/MoNa/MoNA-filtered-curated-with-maccs-and-structure-onehot.json")
DATA_PATH = Path("Data Processing/MoNa/MoNA-filtered-curated-with-maccs-and-structure.json")
SCHEMA_PATH = Path("Data Processing/MoNa/feature_vector_mixed_schema.json")
OUTPUT_PATH = Path("models/multitask_mlp_hardcoded.pt")

SPECTRUM_LEN = 1000
HIDDEN_DIMS: List[int] = [512, 256, 128]
DROPOUT = 0.3
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15


def run_hardcoded_training() -> None:
    # Load schema
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    maccs_length = int(schema.get("maccs_length", 167))

    # Build dataset (compact one-hot packed format)
    #dataset = OneHotDecodedDataset(
        #data_path=DATA_PATH,
        #schema_path=SCHEMA_PATH,
        #spectrum_len=SPECTRUM_LEN,
    #)
    dataset = MixedFeatureDataset(
        data_path=DATA_PATH,
        schema_path=SCHEMA_PATH
    )

    print(f"Records: {len(dataset)}")
    print(f"Onehot features: {dataset.n_categorical}")
    print(f"Continuous featuers: {dataset.n_continuous}")

    x, targets = dataset[0]

    print(f"Spectrum shape: {x.shape}")
    print(f"Binary target shape: {targets['binary'].shape}")
    print(f"Continuous target shape: {targets['continuous'].shape}")

    # Build categorical_sizes dict in the dataset's feature order to ensure matching heads
    feature_schema = schema["feature_schema"]
    categorical_sizes: Dict[str, int] = {}
    for feat in dataset.feature_order:
        spec = feature_schema.get(feat)
        if spec[feat]['type'] != 'onehot':
            continue
        if spec is None:
            raise ValueError(f"Feature {feat} missing from schema.feature_schema")
        categorical_sizes[feat] = len(spec["onehot_names"])

    print(f"Categorical sizes: {categorical_sizes}")

    # Construct model
    model = MultitaskMLPWithQRegression(
        spectrum_len=SPECTRUM_LEN,
        hidden_dims=HIDDEN_DIMS,
        num_binary=maccs_length,
        categorical_sizes=categorical_sizes,
        num_regression=dataset.n_continuous,
        num_quantiles=3,
        dropout=DROPOUT,
    )

    # Split dataset
    train_ds, val_ds, test_ds = split_dataset(dataset, train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION)

    # Auto device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Prepare optimizer, loss fns and dataloaders
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    binary_loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
    categorical_loss_fn = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Checkpoint/resume support
    # If OUTPUT_PATH exists and contains epoch, resume from next epoch.
    start_epoch = 1
    checkpoint_path_latest = OUTPUT_PATH.with_suffix('.latest.pt')
    if checkpoint_path_latest.exists():
        print(f"Found checkpoint {checkpoint_path_latest}, attempting to resume...")
        ckpt = torch.load(checkpoint_path_latest, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"]) 
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", {}))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resuming from epoch {start_epoch}")

    best_val_loss = float("inf")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, EPOCHS + 1):
        # Training loop (inline to ensure correct signature/behavior)
        model.train()
        total_loss = 0.0
        for spectra, targets in train_loader:
            spectra = spectra.to(device)
            targets = {
                "binary": targets["binary"].to(device),
                "categorical": [t.to(device) for t in targets["categorical"]],
                "continuous": targets["continuous"].to(device),
            }
            optimizer.zero_grad()
            outputs = model(spectra)
            loss = compute_loss(outputs, targets, binary_loss_fn, categorical_loss_fn)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(spectra)

        train_loss = total_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, binary_loss_fn, categorical_loss_fn, device)

        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_binary_acc={val_metrics['binary_accuracy']:.4f} | "
            f"val_cat_acc={val_metrics['categorical_accuracy']:.4f}"
        )

        # Save latest checkpoint
        latest_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "categorical_sizes": categorical_sizes,
            "maccs_length": maccs_length,
        }
        torch.save(latest_ckpt, checkpoint_path_latest)

        # Save best model separately
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "categorical_sizes": categorical_sizes,
                "maccs_length": maccs_length,
            }, OUTPUT_PATH)
            print(f"Saved best model to {OUTPUT_PATH}")

    print("Training complete.")

def main():
    # Load schema
    logger.info(f"Loading schema from {SCHEMA_PATH}...")
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)

    maccs_length = int(schema.get("maccs_length", 167))

    # Build dataset (compact one-hot packed format)
    logger.info(f"Building dataset from {DATA_PATH}...")
    dataset = MixedRawFeatureDataset(
        data_path=DATA_PATH,
        schema_path=SCHEMA_PATH
    )

    logger.info(f"Dataset records: {len(dataset)}")
    logger.info(f"Onehot features: {dataset.n_categorical}")
    logger.info(f"Continuous features: {dataset.n_continuous}")

    x, targets = dataset[0]
    logger.info(f"Spectrum shape: {x.shape}")
    logger.info(f"Continuous target shape: {targets['continuous'].shape}")

    # Build categorical_sizes dict in the dataset's feature order to ensure matching heads
    logger.info("Deriving categorical_sizes from dataset and schema...")
    feature_schema = schema["feature_schema"]
    categorical_sizes: Dict[str, int] = {}
    for feat in dataset.onehot_feature_order:
        spec = feature_schema.get(feat)
        if spec['type'] != 'onehot':
            continue
        if spec is None:
            raise ValueError(f"Feature {feat} missing from schema.feature_schema")
        categorical_sizes[feat] = len(spec["onehot_names"])

    # Construct model
    logger.info("Constructing model...")
    model = MultitaskMLPWithQRegression(
        spectrum_len=SPECTRUM_LEN,
        hidden_dims=HIDDEN_DIMS,
        num_binary=maccs_length,
        categorical_sizes=categorical_sizes,
        num_regression=dataset.n_continuous,
        num_quantiles=3,
        dropout=DROPOUT,
    )

    # Split dataset
    logger.info("Splitting dataset into train/val/test...")
    train_ds, val_ds, test_ds = split_dataset(dataset, train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION)
    # cast to MixedRawFeatureDataset to access normalisation
    train_ds = MixedRawFeatureDataset.from_subset(train_ds)
    val_ds = MixedRawFeatureDataset.from_subset(val_ds)
    test_ds = MixedRawFeatureDataset.from_subset(test_ds)

    # Export raw test set records to a JSON file next to the model output (for offline evaluation)
    try:
        test_out = OUTPUT_PATH.with_suffix('.test.json')
        logger.info(f"Exporting test set ({len(test_ds)} records) to {test_out}...")
        with open(test_out, 'w', encoding='utf-8') as fh:
            fh.write('[')
            for i in range(len(test_ds)):
                # MixedRawFeatureDataset supports _load_record (raw dict) via index-based loading
                if hasattr(test_ds, '_load_record'):
                    rec = test_ds._load_record(i)
                else:
                    # fallback for older in-memory implementations
                    rec = test_ds.records[i]
                fh.write(json.dumps(rec))
                if i != len(test_ds) - 1:
                    fh.write(',')
            fh.write(']')
        logger.info(f"Wrote test set to {test_out}")
    except Exception as exc:
        logger.warning(f"Failed to export test set: {exc}")

    # Normalise
    logger.info("Normalising continuous features...")
    mean, std = train_ds.normalise_continuous()
    _, _ = val_ds.normalise_continuous(mean, std)
    _, _ = test_ds.normalise_continuous(mean, std)
    # Save mean and std for later use in inference
    normalisation_stats = {
        "mean": mean,
        "std": std,
    }
    with open(OUTPUT_PATH.with_suffix('.normalisation.json'), "w", encoding="utf-8") as f:
        json.dump(normalisation_stats, f)

    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Test samples: {len(test_ds)}")
    train(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        n_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LR,
        output_path=OUTPUT_PATH,
    )


if __name__ == "__main__":
    #run_hardcoded_training()
    main()
