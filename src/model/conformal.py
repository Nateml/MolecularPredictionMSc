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


def compute_calibration_scores(
    model,
    calibration_loader: DataLoader,
    maccs_only: bool = False,
    quantile_low_idx: int = 0,
    quantile_high_idx: int = 4,
) -> dict:
    """Run inference on the calibration set and return nonconformity scores.

    The returned dict can be cached and passed to thresholds_from_scores() to
    compute thresholds for any alpha without re-running inference.

    Returns:
        {
            "binary": (N, num_binary) nonconformity scores,
            "categorical": list of (N,) scores per categorical feature,
            "quantile": (N, F) scores or None,
            "quantile_low_idx": int,
            "quantile_high_idx": int,
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

            if not maccs_only:
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
    if not maccs_only:
        all_categorical_logits = [torch.cat(l) for l in all_categorical_logits]
        all_categorical_targets = [torch.cat(t) for t in all_categorical_targets]
    all_quantile_outputs = torch.cat(all_quantile_outputs) if all_quantile_outputs else None
    all_quantile_targets = torch.cat(all_quantile_targets) if all_quantile_targets else None

    scores_binary = binary_score_fn(all_binary_logits, all_binary_targets)
    scores_categorical = (
        categorical_score_fn(all_categorical_logits, all_categorical_targets)
        if not maccs_only else []
    )
    scores_quantile = (
        quantile_score_fn(
            all_quantile_outputs, all_quantile_targets,
            trained_quantiles=[],
            low_idx=quantile_low_idx, high_idx=quantile_high_idx,
        )
        if (not maccs_only and all_quantile_outputs is not None) else None
    )

    return {
        "binary": scores_binary,
        "categorical": scores_categorical,
        "quantile": scores_quantile,
        "quantile_low_idx": quantile_low_idx,
        "quantile_high_idx": quantile_high_idx,
    }


def thresholds_from_scores(scores: dict, alpha: float, joint: bool = False) -> dict:
    """Compute conformal thresholds from precomputed nonconformity scores.

    Args:
        scores: dict returned by compute_calibration_scores()
        alpha: miscoverage level; target coverage = 1 - alpha
        joint: if True, apply rank-based joint calibration

    Returns:
        {
            "binary_thresholds": (num_binary,),
            "categorical_thresholds": list of scalars,
            "quantile_thresholds": (F,) or None,
        }
    """
    scores_binary = scores["binary"]
    scores_categorical = scores.get("categorical", [])
    scores_quantile = scores.get("quantile", None)

    n = scores_binary.shape[0]
    level = min((1 - alpha) * (1 + 1 / n), 1.0)

    q_hat_binary = torch.quantile(scores_binary, level, dim=0, interpolation="higher")
    q_hat_categorical = [
        torch.quantile(s, level, interpolation="higher") for s in scores_categorical
    ]
    q_hat_quantile = (
        torch.quantile(scores_quantile, level, dim=0, interpolation="higher").clamp(min=0.0)
        if scores_quantile is not None else None
    )

    if joint:
        def _empirical_ranks(scores: torch.Tensor) -> torch.Tensor:
            """Empirical CDF rank per column. (N, K) -> (N, K) in (0, 1]."""
            N = scores.shape[0]
            order = scores.argsort(dim=0)
            ranks = torch.empty_like(scores)
            ranks.scatter_(
                0, order,
                (torch.arange(1, N + 1, dtype=scores.dtype) / N)
                .unsqueeze(1).expand_as(scores),
            )
            return ranks

        parts: list[torch.Tensor] = []
        parts.append(_empirical_ranks(scores_binary).max(dim=1).values)

        if scores_categorical:
            cat_scores = torch.stack(scores_categorical, dim=1)
            parts.append(_empirical_ranks(cat_scores).max(dim=1).values)

        if scores_quantile is not None:
            parts.append(_empirical_ranks(scores_quantile).max(dim=1).values)

        joint_score = torch.stack(parts, dim=1).max(dim=1).values
        T = torch.quantile(joint_score, level, interpolation="higher")

        q_hat_binary = torch.quantile(scores_binary, T, dim=0, interpolation="higher")
        q_hat_categorical = [
            torch.quantile(s, T, interpolation="higher") for s in scores_categorical
        ]
        if scores_quantile is not None:
            q_hat_quantile = torch.quantile(scores_quantile, T, dim=0, interpolation="higher").clamp(min=0.0)

        print(f"Joint calibration: effective level T = {T.item():.4f} (marginal level was {level:.4f})")

    return {
        "binary_thresholds": q_hat_binary,
        "categorical_thresholds": q_hat_categorical,
        "quantile_thresholds": q_hat_quantile,
    }


def calibrate_model(
    model,
    calibration_loader: DataLoader,
    alpha: float,
    quantiles: list[float] = [0.025, 0.05, 0.5, 0.95, 0.975],
    quantile_low_idx: int = 0,
    quantile_high_idx: int = 4,
    joint: bool = False,
    maccs_only: bool = False,
) -> dict:
    """Compute conformal prediction thresholds from a calibration set.

    Convenience wrapper around compute_calibration_scores + thresholds_from_scores.
    Prefer calling those two functions directly when running multiple alpha values,
    so that inference runs only once.
    """
    scores = compute_calibration_scores(
        model, calibration_loader,
        maccs_only=maccs_only,
        quantile_low_idx=quantile_low_idx,
        quantile_high_idx=quantile_high_idx,
    )
    return thresholds_from_scores(scores, alpha=alpha, joint=joint)


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
    if thresholds["categorical_thresholds"]:
        for logits_i, q_hat_i in zip(outputs["categorical"], thresholds["categorical_thresholds"]):
            scores = -F.log_softmax(logits_i, dim=-1)
            include = scores <= q_hat_i.to(logits_i.device)
            empty = ~include.any(dim=-1)
            if empty.any():
                argmax_idx = logits_i.argmax(dim=-1)
                rows = torch.nonzero(empty, as_tuple=False).squeeze(-1)
                include[rows, argmax_idx[rows]] = True
            categorical_sets.append(include)
    else:
        for logits_i in outputs["categorical"]:
            B, K = logits_i.shape
            include = torch.zeros(B, K, dtype=torch.bool, device=logits_i.device)
            include.scatter_(1, logits_i.argmax(dim=-1, keepdim=True), True)
            categorical_sets.append(include)
    results["categorical"] = categorical_sets

    # Quantile
    results["quantile"] = None
    if "quantile" in outputs and thresholds["quantile_thresholds"] is not None:
        q_outputs = torch.stack(outputs["quantile"], dim=1)  # (B, F, Q)
        q_hat_q = thresholds["quantile_thresholds"].to(q_outputs.device).clamp(min=0.0)
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
    num_quantiles = model.num_quantiles
    quantile_low_idx = 0
    quantile_high_idx = num_quantiles - 1
    with torch.no_grad():
        outputs = model(spectra.to(device))
    return sets_from_outputs(outputs, thresholds, quantile_low_idx, quantile_high_idx)
