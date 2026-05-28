from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def aps_score_fn(logits_list, targets_list):
    """
    Adaptive prediction sets (APS) score function for categorical features.

    Args:
        logits_list: list of (B, K_i) raw model outputs for each categorical feature
        targets_list: list of (B,) true labels for each categorical feature
    Returns:
        list of (B,) nonconformity scores for each categorical feature
    """
    scores = []
    for logits_i, targets_i in zip(logits_list, targets_list):
        probs = F.softmax(logits_i, dim=-1)  # (B, K_i)
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        true_class_idx = (sorted_indices == targets_i.unsqueeze(-1)).nonzero(as_tuple=True)[1]  # (B,)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # (B, K_i)
        score = cumsum_probs[torch.arange(logits_i.size(0)), true_class_idx]  # (B,)
        scores.append(score)
    return scores


def binary_score_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Nonconformity scores for binary classification: negative log-prob of true class.

    Args:
        logits: (batch_size, num_binary) raw logits before sigmoid
        targets: (batch_size, num_binary) binary labels
    Returns:
        (batch_size, num_binary) nonconformity scores
    """
    log_probs = F.logsigmoid(logits) * targets + F.logsigmoid(-logits) * (1 - targets)
    return -log_probs


def categorical_score_fn(logits_list, targets_list):
    """
    Nonconformity scores for categorical features: negative log-softmax of true class.

    Args:
        logits_list: list of (B, K_i)
        targets_list: list of (B,)
    Returns:
        list of (B,) scores per feature
    """
    scores = []
    for logits_i, targets_i in zip(logits_list, targets_list):
        log_probs = F.log_softmax(logits_i, dim=-1)  # (B, K_i)
        s = -log_probs[torch.arange(logits_i.size(0)), targets_i]
        scores.append(s)
    return scores


def quantile_score_fn(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    trained_quantiles: list[float],
    low_idx: int,
    high_idx: int,
) -> torch.Tensor:
    """
    Nonconformity scores for quantile regression: max distance outside [q_low, q_high].

    Args:
        outputs: (B, F, Q) predicted quantiles per regression feature
        targets: (B, F) true values
        trained_quantiles: quantile levels the model was trained on (e.g. [0.1, 0.5, 0.9])
        low_idx: index into Q for lower bound
        high_idx: index into Q for upper bound
    Returns:
        (B, F) per-sample per-feature nonconformity scores
    """
    targets = targets.unsqueeze(-1)  # (B, F, 1)
    q_low = outputs[:, :, low_idx:low_idx + 1]    # (B, F, 1)
    q_high = outputs[:, :, high_idx:high_idx + 1]  # (B, F, 1)
    scores = torch.maximum(q_low - targets, targets - q_high)  # (B, F, 1)
    return scores.squeeze(-1)  # (B, F)


def calibrate_model(
    model,
    calibration_loader: DataLoader,
    alpha: float,
    quantiles: list[float] = [0.025, 0.05, 0.5, 0.95, 0.975],
    quantile_low_idx: int = 0,
    quantile_high_idx: int = 4,
) -> dict:
    """
    Compute conformal prediction thresholds from a calibration set.

    Expects DataLoader batches of (spectra, targets) where targets contains:
        "binary": (B, num_binary)
        "categorical": list of (B,) per feature
        "continuous": (B, F) normalized continuous values (optional; needed for quantile thresholds)

    Returns:
        {
            "binary_thresholds": (num_binary,),
            "categorical_thresholds": list of scalars (one per categorical feature),
            "quantile_thresholds": (F,) or None,
        }
    """
    model.eval()
    device = next(model.parameters()).device

    all_binary_logits, all_binary_targets = [], []
    all_quantile_outputs, all_quantile_targets = [], []
    all_categorical_logits = None
    all_categorical_targets = None

    with torch.no_grad():
        for spectra, targets in calibration_loader:
            spectra = spectra.to(device)
            cont = targets.get("continuous", None)
            binary_targets = targets["binary"].to(device)
            cat_targets = [t.to(device) for t in targets["categorical"]]
            reg_targets = cont.to(device) if cont is not None else None

            outputs = model(spectra)

            all_binary_logits.append(outputs["binary"].cpu())
            all_binary_targets.append(binary_targets.cpu())

            if all_categorical_logits is None:
                n_cat = len(outputs["categorical"])
                all_categorical_logits = [[] for _ in range(n_cat)]
                all_categorical_targets = [[] for _ in range(n_cat)]
            for i, (logits_i, target_i) in enumerate(zip(outputs["categorical"], cat_targets)):
                all_categorical_logits[i].append(logits_i.cpu())
                all_categorical_targets[i].append(target_i.cpu())

            if "quantile" in outputs and reg_targets is not None:
                q_out = torch.stack(outputs["quantile"], dim=1)  # (B, F, Q)
                all_quantile_outputs.append(q_out.cpu())
                all_quantile_targets.append(reg_targets.cpu())

    all_binary_logits = torch.cat(all_binary_logits)
    all_binary_targets = torch.cat(all_binary_targets)
    all_categorical_logits = [torch.cat(l) for l in all_categorical_logits]
    all_categorical_targets = [torch.cat(t) for t in all_categorical_targets]
    all_quantile_outputs = torch.cat(all_quantile_outputs) if all_quantile_outputs else None
    all_quantile_targets = torch.cat(all_quantile_targets) if all_quantile_targets else None

    scores_binary = binary_score_fn(all_binary_logits, all_binary_targets)
    scores_categorical = categorical_score_fn(all_categorical_logits, all_categorical_targets)
    scores_quantile = (
        quantile_score_fn(
            all_quantile_outputs, all_quantile_targets,
            trained_quantiles=quantiles,
            low_idx=quantile_low_idx, high_idx=quantile_high_idx,
        )
        if all_quantile_outputs is not None else None
    )

    n = scores_binary.shape[0]
    level = min((1 - alpha) * (1 + 1 / n), 1.0)

    q_hat_binary = torch.quantile(scores_binary, level, dim=0, interpolation="higher")
    q_hat_categorical = [
        torch.quantile(s, level, interpolation="higher") for s in scores_categorical
    ]
    q_hat_quantile = (
        torch.quantile(scores_quantile, level, dim=0, interpolation="higher")
        if scores_quantile is not None else None
    )

    return {
        "binary_thresholds": q_hat_binary,
        "categorical_thresholds": q_hat_categorical,
        "quantile_thresholds": q_hat_quantile,
    }


def sets_from_outputs(
    outputs: dict,
    thresholds: dict,
    quantile_low_idx: int = 0,
    quantile_high_idx: int = 4,
) -> dict:
    """
    Compute conformal prediction sets from precomputed model outputs.

    Args:
        outputs: dict with "binary", "categorical", and optionally "quantile" keys
        thresholds: output of calibrate_model()
        quantile_low_idx: index of lower quantile in the Q dimension
        quantile_high_idx: index of upper quantile in the Q dimension

    Returns:
        {
            "binary": (B, num_binary, 2) bool — last dim indexes class {0, 1},
            "categorical": list of (B, K_i) bool,
            "quantile": (B, F, 2) float in normalized space, or None,
        }
    """
    results = {}

    # Binary
    logits = outputs["binary"]  # (B, num_binary)
    q_hat = thresholds["binary_thresholds"].to(logits.device)
    s1 = -F.logsigmoid(logits)   # score if y=1
    s0 = -F.logsigmoid(-logits)  # score if y=0
    include_0 = s0 <= q_hat
    include_1 = s1 <= q_hat
    empty = ~(include_0 | include_1)
    pred_is_1 = logits >= 0
    include_0 = include_0 | (empty & ~pred_is_1)
    include_1 = include_1 | (empty & pred_is_1)
    results["binary"] = torch.stack([include_0, include_1], dim=-1)  # (B, num_binary, 2)

    # Categorical
    categorical_sets = []
    for logits_i, q_hat_i in zip(outputs["categorical"], thresholds["categorical_thresholds"]):
        scores = -F.log_softmax(logits_i, dim=-1)
        include = scores <= q_hat_i.to(logits_i.device)
        empty = ~include.any(dim=-1)
        if empty.any():
            argmax_idx = logits_i.argmax(dim=-1)
            rows = torch.nonzero(empty, as_tuple=False).squeeze(-1)
            include[rows, argmax_idx[rows]] = True
        categorical_sets.append(include)
    results["categorical"] = categorical_sets

    # Quantile
    results["quantile"] = None
    if "quantile" in outputs and thresholds["quantile_thresholds"] is not None:
        q_outputs = torch.stack(outputs["quantile"], dim=1)  # (B, F, Q)
        q_hat_q = thresholds["quantile_thresholds"].to(q_outputs.device)
        q_low = q_outputs[:, :, quantile_low_idx]
        q_high = q_outputs[:, :, quantile_high_idx]
        results["quantile"] = torch.stack([q_low - q_hat_q, q_high + q_hat_q], dim=-1)  # (B, F, 2)

    return results


def predict_sets(
    model,
    spectra: torch.Tensor,
    thresholds: dict,
    quantile_low_idx: int = 0,
    quantile_high_idx: int = 2,
) -> dict:
    """
    Run model forward pass then compute conformal prediction sets.

    Returns same format as sets_from_outputs().
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        outputs = model(spectra.to(device))
    return sets_from_outputs(outputs, thresholds, quantile_low_idx, quantile_high_idx)
