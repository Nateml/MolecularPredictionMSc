from pathlib import Path

from neural_net.model import MultitaskMLP
from torch.utils.data import DataLoader, Subset
from torch import nn
from neural_net.onehot_decoded_dataset import OneHotDecodedDataset

import torch
import torch.nn.functional as F
import argparse

from neural_net.train import compute_loss, split_dataset

def binary_score_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute nonconformity scores for binary classification using the negative log-probability of the true class.

    Args:
        logits: Tensor of shape (batch_size, num_binary) containing raw model output for binary features (before sigmoid).
        targets: Tensor of shape (batch_size, num_binary) containing true binary labels (0 or 1).
    Returns:
        Tensor of shape (batch_size, num_binary) containing nonconformity scores (lower is more conforming).
    """

    # Using binary cross-entropy loss as the nonconformity score
    log_probs = F.logsigmoid(logits) * targets + F.logsigmoid(-logits) * (1 - targets)
    return -log_probs  # Negate to get nonconformity score (lower is better)
    
def categorical_score_fn(logits_list, targets_list):
    """
    logits_list: list of (B, K_i)
    targets_list: list of (B,)
    
    returns:
        list of (B,) scores per feature
    """
    scores = []

    for logits_i, targets_i in zip(logits_list, targets_list):
        log_probs = F.log_softmax(logits_i, dim=-1)  # (B, K_i)
        s = -log_probs[torch.arange(logits_i.size(0)), targets_i]
        scores.append(s)

    return scores  # list of (B,)

def aps_score_fn(logits_list, targets_list):
    """
    Adaptive prediciton sets (APS) score function for categorical features.

    Args:
        logits_list: list of (B, K_i) raw model outputs for each categorical feature
        targets_list: list of (B,) true labels for each categorical feature
    Returns:
        list of (B,) nonconformity scores for each categorical feature
    """

    scores = []

    for logits_i, targets_i in zip(logits_list, targets_list):
        probs = F.softmax(logits_i, dim=-1)  # (B, K_i)
        
        # Sort the probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

        # Find the index of the true class in sorted order
        true_class_idx = (sorted_indices == targets_i.unsqueeze(-1)).nonzero(as_tuple=True)[1]  # (B,)

        # Cumulutive sum 
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, K_i)

        # The score is the cumulative probability up to and including the true class
        score = cumsum_probs[torch.arange(logits_i.size(0)), true_class_idx]  # (B,)
        scores.append(score)

    return scores # list of (B,)


def quantile_score_fn(outputs: torch.Tensor, targets: torch.Tensor, trained_quantiles: list[float], low_idx: float, high_idx: float) -> torch.Tensor:
    """
    Compute nonconformity scores for regression using quantile regression outputs.

    Args:
        outputs: Tensor of shape (batch_size, num_regression_features, num_quantiles) containing predicted quantiles for each regression feature.
        targets: Tensor of shape (batch_size, num_regression_features) containing true target values for each regression feature.
        trained_quantiles: List of quantiles that the model was trained to predict (e.g. [0.1, 0.5, 0.9]).
        low_idx: Index of the quantile in trained_quantiles to use as the lower threshold (e.g. 0 for 0.1).
        high_idx: Index of the quantile in trained_quantiles to use as the upper threshold (e.g. 2 for 0.9).
    """

    # Expand targets to match quantile dimension
    targets = targets.unsqueeze(-1)  # (B, F, 1)

    # Select quantiles along last dimension
    q_low = outputs[:, :, low_idx:low_idx+1]   # (B, F, 1)
    q_high = outputs[:, :, high_idx:high_idx+1]  # (B, F, 1)

    # Compute per-feature nonconformity
    scores = torch.maximum(q_low - targets, targets - q_high)  # (B, F, 1)

    # Remove last dim
    scores = scores.squeeze(-1)  # (B, F)

    # Aggregate across features → scalar per sample
    return scores


def calibrate_model(
    model,
    calibration_loader,
    alpha: float,
    quantiles: list[float] = [0.1, 0.5, 0.9],
    quantile_low_idx: int = 0,
    quantile_high_idx: int = 2,
):
    model.eval()
    device = next(model.parameters()).device

    # --------------------------
    # Storage
    # --------------------------
    all_binary_logits, all_binary_targets = [], []
    all_quantile_outputs, all_quantile_targets = [], []

    # We initialize categorical storage lazily (we don’t know C yet)
    all_categorical_logits = None
    all_categorical_targets = None

    # --------------------------
    # Collect calibration data
    # --------------------------
    with torch.no_grad():
        for spectra, targets in calibration_loader:
            spectra = spectra.to(device)

            targets = {
                "binary": targets["binary"].to(device),
                "categorical": [t.to(device) for t in targets["categorical"]],
                "regression": targets.get("regression", None).to(device)
                if targets.get("regression", None) is not None else None,
            }

            outputs = model(spectra)

            # -------- Binary --------
            all_binary_logits.append(outputs["binary"].cpu())
            all_binary_targets.append(targets["binary"].cpu())

            # -------- Categorical (list-based) --------
            if all_categorical_logits is None:
                num_cat_features = len(outputs["categorical"])
                all_categorical_logits = [[] for _ in range(num_cat_features)]
                all_categorical_targets = [[] for _ in range(num_cat_features)]

            for i, (logits_i, target_i) in enumerate(
                zip(outputs["categorical"], targets["categorical"])
            ):
                all_categorical_logits[i].append(logits_i.cpu())   # (B, K_i)
                all_categorical_targets[i].append(target_i.cpu())  # (B,)

            # -------- Regression --------
            if "regression" in outputs and targets["regression"] is not None:
                q_out = torch.stack(outputs["regression"], dim=1)  # (B, F, Q)
                all_quantile_outputs.append(q_out.cpu())
                all_quantile_targets.append(targets["regression"].cpu())

    # --------------------------
    # Concatenate
    # --------------------------
    all_binary_logits = torch.cat(all_binary_logits)
    all_binary_targets = torch.cat(all_binary_targets)

    all_categorical_logits = [
        torch.cat(l) for l in all_categorical_logits
    ]
    all_categorical_targets = [
        torch.cat(t) for t in all_categorical_targets
    ]

    all_quantile_outputs = (
        torch.cat(all_quantile_outputs) if all_quantile_outputs else None
    )
    all_quantile_targets = (
        torch.cat(all_quantile_targets) if all_quantile_targets else None
    )

    # --------------------------
    # Compute scores
    # --------------------------

    # Binary
    scores_binary = binary_score_fn(all_binary_logits, all_binary_targets)  # (B, num_binary)

    # Categorical (per feature)
    scores_categorical = aps_score_fn(all_categorical_logits, all_categorical_targets)  # list of (B,) per feature

    # Quantile
    scores_quantile = (
        quantile_score_fn(
            all_quantile_outputs,
            all_quantile_targets,
            trained_quantiles=quantiles,
            low_idx=quantile_low_idx,
            high_idx=quantile_high_idx,
        )
        if all_quantile_outputs is not None
        else None
    )

    # --------------------------
    # Compute thresholds
    # --------------------------

    # Binary (per feature)
    q_hat_binary = torch.quantile(scores_binary, 1 - alpha, dim=0)

    # Categorical (per feature → list)
    q_hat_categorical = [
        torch.quantile(s, 1 - alpha) for s in scores_categorical
    ]

    # Quantile (per regression feature)
    q_hat_quantile = (
        torch.quantile(scores_quantile, 1 - alpha, dim=0)
        if scores_quantile is not None else None
    )

    return {
        "binary_thresholds": q_hat_binary,                 # (num_binary,)
        "categorical_thresholds": q_hat_categorical,       # list of scalars
        "quantile_thresholds": q_hat_quantile,             # (num_reg_features,) or None
    }


def predict_sets(
    model,
    spectra: torch.Tensor,
    thresholds: dict,
    quantiles: list[float] = [0.1, 0.5, 0.9],
    quantile_low_idx: int = 0,
    quantile_high_idx: int = 2,
):
    """
    Generate conformal prediction sets at test time.

    Args:
        model: trained model
        spectra: (B, input_dim)
        thresholds: output of calibrate_model()
    
    Returns:
        dict with:
            - binary: list of (B, num_binary, 2) boolean tensors
            - categorical: list of (B, num_cat_features, num_classes) boolean tensors
            - regression: (lower, upper) each (B, num_reg_features)
    """

    device = next(model.parameters()).device
    spectra = spectra.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(spectra)

    results = {}

    # ======================
    # 1. Binary prediction sets
    # ======================
    logits = outputs["binary"]  # (B, num_binary)
    q_hat = thresholds["binary_thresholds"].to(logits.device)  # (num_binary,)

    probs = torch.sigmoid(logits)

    # Scores for both possible labels
    s0 = -torch.log(1 - probs)   # score if y=0
    s1 = -torch.log(probs)       # score if y=1

    include_0 = s0 <= q_hat      # (B, num_binary)
    include_1 = s1 <= q_hat

    # Stack into final set: last dim = {0,1}
    binary_sets = torch.stack([include_0, include_1], dim=-1)  # (B, num_binary, 2)

    results["binary"] = binary_sets

    # ======================
    # 2. Categorical prediction sets
    # ======================
    cat_outputs = outputs["categorical"]  # list of length C, each (B, K_i)
    q_hat_cat = thresholds["categorical_thresholds"]

    # Ensure thresholds are iterable per feature
    if isinstance(q_hat_cat, torch.Tensor):
        q_hat_cat = q_hat_cat.tolist()

    categorical_sets = []

    for logits_i, q_hat_i in zip(cat_outputs, q_hat_cat):
        logits_i = logits_i.to(device)  # (B, K_i)
        q_hat_i = torch.tensor(q_hat_i, device=logits_i.device)

        log_probs = F.log_softmax(logits_i, dim=-1)  # (B, K_i)
        scores = -log_probs

        pred_set_i = scores <= q_hat_i  # (B, K_i)
        categorical_sets.append(pred_set_i)

    results["categorical"] = categorical_sets  # list of (B, K_i)

    # Broadcast thresholds: (C,) → (B, C, K)
    prediction_sets_cat = scores <= q_hat_cat.unsqueeze(0).unsqueeze(-1)

    results["categorical"] = prediction_sets_cat  # (B, C, K)

    # ======================
    # 3. Regression (quantile) intervals
    # ======================
    if "regression" in outputs and thresholds["quantile_thresholds"] is not None:
        q_outputs = torch.stack(outputs["regression"], dim=1)  # (B, F, Q)
        q_hat_q = thresholds["quantile_thresholds"].to(q_outputs.device)  # (F,)

        q_low = q_outputs[:, :, quantile_low_idx]   # (B, F)
        q_high = q_outputs[:, :, quantile_high_idx]

        lower = q_low - q_hat_q
        upper = q_high + q_hat_q

        results["regression"] = (lower, upper)

    return results

def main(args: list[str]):
    # Hardcoded values for now
    DATA_PATH = Path("Data Processing/MoNa/MoNA-filtered-curated-with-maccs-and-structure-onehot.json")
    SCHEMA_PATH = Path("Data Processing/MoNa/feature_vector_onehot_schema.json")
    OUTPUT_PATH = Path("models/multitask_mlp_hardcoded.pt")
    SPECTRUM_LEN = 1000
    DROPOUT = 0.3
    alpha = 0.05

    # Load model
    ckpt = torch.load(args.model_checkpoint, map_location="cpu")
    model = MultitaskMLP(
        spectrum_len=ckpt["spectrum_len"],
        hidden_dims=ckpt["hidden_dims"],
        num_binary=ckpt["num_binary"],
        categorical_sizes=ckpt["categorical_sizes"],
        dropout=DROPOUT,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load calibration data
    dataset = OneHotDecodedDataset(
        data_path=args.calibration_data,
        schema_path=SCHEMA_PATH,
        spectrum_len=SPECTRUM_LEN,
    )

    _, _, test_ds = split_dataset(dataset, train_fraction=0.7, val_fraction=0.15)

    # Split test_ds into calibration and test sets
    calib_size = int(0.2 * len(test_ds))
    calib_ds = Subset(test_ds, list(range(calib_size)))
    test_ds = Subset(test_ds, list(range(calib_size, len(test_ds))))

    calibration_loader = DataLoader(calib_ds, batch_size=64, shuffle=False)

    # Calibrate model
    thresholds = calibrate_model(model, calibration_loader, alpha)

    # Save thresholds
    torch.save(thresholds, args.output_thresholds)

if __name__ == "__main__":
    main()