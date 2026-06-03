"""Batch and single-spectrum inference for MultitaskMLP and MultitaskMLPWithQRegression.

Usage
-----
    # Single spectrum tensor:
    pred = predict_single(x, model, dataset, thresholds=thresholds)
    print(pred)

    # Batch over dataset records:
    results = run_inference(dataset, model, indices=[0, 1, 2], thresholds=thresholds)
    for pred in results:
        for combo in itertools.product(*[range(lo, hi+1) for lo, hi in pred.count_ranges.values()]):
            counts = dict(zip(pred.count_ranges, combo))
            interface = PredictionInterface.model_validate({**pred.structural, **counts})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from data.datasets import RawFeatureDataset, SpectrumDataset
from model.conformal import sets_from_outputs
from predicates.definitions import MACCS_DEFINITIONS, FEATURE_VECTOR_DEFINITIONS
from predicates.interface import PredictionEntry, ContinuousRange, PredictionInterface

_ELEMENT_TO_COUNT_KEY: dict[str, str] = {
    "element_C": "num_carbons",
    "element_H": "num_hydrogens",
    "element_N": "num_nitrogens",
    "element_O": "num_oxygens",
    "element_P": "num_phosphorus",
    "element_S": "num_sulfurs",
}

# Ordered list so count_ranges dicts have consistent key order.
_ALL_COUNT_KEYS: list[str] = list(_ELEMENT_TO_COUNT_KEY.values())

_PREDICATE_TO_FVD_KEY: dict[str, int] = {
    v["predicate"]: k for k, v in FEATURE_VECTOR_DEFINITIONS.items()
}


@dataclass
class RecordPrediction:
    """Predictions for one record, ready for PredictionInterface construction.

    structural:        {key_id: PredictionEntry} — binary MACCS + categorical features,
                       confidence-filtered. Keys absent when prediction is uncertain.
    count_ranges:      {atom_count_key: (lo, hi)} — inclusive integer ranges per element.
                       All six keys always present; (0, 0) when unavailable.
    continuous_ranges: {feature_name: (lo_float, hi_float)} — denormalized float ranges
                       for all continuous features. Used to generate extra ASP constraints
                       for non-element structural features (ring_count, mass, etc.).

    To build a PredictionInterface:
        for combo in itertools.product(*[range(lo, hi+1) for lo, hi in pred.count_ranges.values()]):
            counts = dict(zip(pred.count_ranges, combo))
            interface = PredictionInterface.model_validate({**pred.structural, **counts})
    """

    structural: dict
    count_ranges: dict[str, tuple[int, int]]
    continuous_ranges: dict[str, tuple[float, float]] = field(default_factory=dict)

    def __str__(self) -> str:
        lines = ["RecordPrediction"]

        lines.append("  counts:")
        for key, (lo, hi) in self.count_ranges.items():
            if lo == hi:
                lines.append(f"    {key}: {lo}")
            else:
                lines.append(f"    {key}: [{lo}, {hi}]")

        if self.continuous_ranges:
            non_elem = {f: r for f, r in self.continuous_ranges.items()
                        if f not in _ELEMENT_TO_COUNT_KEY}
            if non_elem:
                lines.append(f"  continuous ({len(non_elem)} features):")
                for feature, (lo, hi) in non_elem.items():
                    if round(lo) == round(hi):
                        lines.append(f"    {feature}: {round(lo)}")
                    else:
                        lines.append(f"    {feature}: [{lo:.2f}, {hi:.2f}]")

        lines.append(f"  structural ({len(self.structural)} entries):")
        for key, entry in self.structural.items():
            proba_str = f"  p={entry.proba[0]:.3f}" if entry.proba else ""
            lines.append(f"    [{key}] pred_set={entry.pred_set}{proba_str}")

        return "\n".join(lines)

    def to_interface(self, counts: dict[str, int] | None = None) -> PredictionInterface:
        """Build a PredictionInterface from this prediction.

        Args:
            counts: Explicit atom counts {e.g. 'num_carbons': 7}. If None, uses
                    the midpoint of each count range.
        """
        if counts is None:
            counts = {k: (lo + hi) // 2 for k, (lo, hi) in self.count_ranges.items()}
        return PredictionInterface.model_validate({
            **{k: v.model_dump() for k, v in self.structural.items()},
            **counts,
            **{f: {"lo": lo, "hi": hi} for f, (lo, hi) in self.continuous_ranges.items()},
        })


# ---------------------------------------------------------------------------
# Private extraction helpers (operate on single-record tensors)
# ---------------------------------------------------------------------------

def _structural_from_threshold(
    binary_logits_i: torch.Tensor,
    cat_logits_list_i: list[torch.Tensor],
    onehot_feature_order: list[str],
    threshold: float,
) -> dict:
    """Threshold-filtered structural entries from raw per-record logits."""
    structural: dict = {}

    for j, p in enumerate(torch.sigmoid(binary_logits_i).tolist()):
        if j not in MACCS_DEFINITIONS:
            continue
        if p >= threshold:
            structural[j] = PredictionEntry(pred_set=[1], proba=[p])
        elif p <= 1.0 - threshold:
            structural[j] = PredictionEntry(pred_set=[0], proba=[1.0 - p])

    for logits_i, feature in zip(cat_logits_list_i, onehot_feature_order):
        cat_probs = F.softmax(logits_i, dim=-1).tolist()
        class_idx = int(torch.argmax(logits_i).item())
        confidence = cat_probs[class_idx]
        if confidence < threshold:
            continue
        predicate_name = f"{feature}_bin_{class_idx}"
        key = _PREDICATE_TO_FVD_KEY.get(predicate_name)
        if key is not None:
            structural[key] = PredictionEntry(pred_set=[1], proba=[confidence])

    return structural


def _structural_from_conf_sets_single(
    binary_sets_i: torch.Tensor,
    cat_sets_list_i: list[torch.Tensor],
    onehot_feature_order: list[str],
) -> dict:
    """Structural entries from conformal prediction sets for a single record.

    Args:
        binary_sets_i:   (num_binary, 2) bool — last dim indexes class {0, 1}
        cat_sets_list_i: list of (K_i,) bool, one per categorical feature
        onehot_feature_order: feature names aligned with cat_sets_list_i
    """
    structural: dict = {}

    for j in range(binary_sets_i.shape[0]):
        if j not in MACCS_DEFINITIONS:
            continue
        inc_0 = binary_sets_i[j, 0].item()
        inc_1 = binary_sets_i[j, 1].item()
        if inc_1 and not inc_0:
            structural[j] = PredictionEntry(pred_set=[1], proba=[1.0])
        elif inc_0 and not inc_1:
            structural[j] = PredictionEntry(pred_set=[0], proba=[1.0])

    for cat_set_i, feature in zip(cat_sets_list_i, onehot_feature_order):
        included = cat_set_i.nonzero(as_tuple=False).squeeze(-1).tolist()
        if len(included) == 1:
            predicate_name = f"{feature}_bin_{included[0]}"
            key = _PREDICATE_TO_FVD_KEY.get(predicate_name)
            if key is not None:
                structural[key] = PredictionEntry(pred_set=[1], proba=[1.0])

    return structural


def _process_quantile_outputs(
    q_logits_list_i: list[torch.Tensor],
    continuous_feature_order: list[str],
    cont_mean: torch.Tensor | None,
    cont_std: torch.Tensor | None,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[float, float]]]:
    """Denormalize quantile outputs into count_ranges and continuous_ranges.

    Uses the outermost quantiles (first and last) as [lo, hi] bounds.

    Returns:
        count_ranges:      {atom_count_key: (lo_int, hi_int)} for element features
        continuous_ranges: {feature_name: (lo_float, hi_float)} for all features
    """
    count_ranges: dict[str, tuple[int, int]] = {k: (0, 0) for k in _ALL_COUNT_KEYS}
    continuous_ranges: dict[str, tuple[float, float]] = {}

    for feat_idx, (q_tensor, feature) in enumerate(zip(q_logits_list_i, continuous_feature_order)):
        q_vals = q_tensor.tolist()
        lo_val, hi_val = q_vals[0], q_vals[-1]
        if cont_mean is not None and cont_std is not None:
            m = cont_mean[feat_idx].item()
            s = cont_std[feat_idx].item()
            lo_val = lo_val * s + m
            hi_val = hi_val * s + m

        continuous_ranges[feature] = (lo_val, hi_val)

        count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
        if count_key is not None:
            lo = max(0, round(lo_val))
            count_ranges[count_key] = (lo, max(lo, round(hi_val)))

    return count_ranges, continuous_ranges


def _process_conf_quantile_sets(
    quantile_sets_i: torch.Tensor,
    continuous_feature_order: list[str],
    cont_mean: torch.Tensor | None,
    cont_std: torch.Tensor | None,
) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[float, float]]]:
    """Denormalize conformal quantile intervals into count_ranges and continuous_ranges.

    Args:
        quantile_sets_i: (F, 2) float — conformal [lo, hi] intervals in normalized space
    """
    count_ranges: dict[str, tuple[int, int]] = {k: (0, 0) for k in _ALL_COUNT_KEYS}
    continuous_ranges: dict[str, tuple[float, float]] = {}

    for feat_idx, feature in enumerate(continuous_feature_order):
        if feat_idx >= quantile_sets_i.shape[0]:
            break
        lo_norm = quantile_sets_i[feat_idx, 0].item()
        hi_norm = quantile_sets_i[feat_idx, 1].item()
        if cont_mean is not None and cont_std is not None:
            m = cont_mean[feat_idx].item()
            s = cont_std[feat_idx].item()
            lo_val = lo_norm * s + m
            hi_val = hi_norm * s + m
        else:
            lo_val, hi_val = lo_norm, hi_norm

        continuous_ranges[feature] = (lo_val, hi_val)

        count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
        if count_key is not None:
            lo = max(0, round(lo_val))
            count_ranges[count_key] = (lo, max(lo, round(hi_val)))

    return count_ranges, continuous_ranges


def _count_ranges_from_categorical(
    cat_logits_list_i: list[torch.Tensor],
    onehot_feature_order: list[str],
) -> dict[str, tuple[int, int]]:
    """Element count ranges (point estimates) from categorical head argmax.

    Used for PackedDataset models where element counts are encoded as
    categorical features (class index == count).
    """
    count_ranges: dict[str, tuple[int, int]] = {k: (0, 0) for k in _ALL_COUNT_KEYS}
    for logits_i, feature in zip(cat_logits_list_i, onehot_feature_order):
        count_key = _ELEMENT_TO_COUNT_KEY.get(feature)
        if count_key is not None:
            val = int(torch.argmax(logits_i).item())
            count_ranges[count_key] = (val, val)
    return count_ranges


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_single(
    x: torch.Tensor,
    model: torch.nn.Module,
    dataset: SpectrumDataset | Subset,
    threshold: float = 0.5,
    thresholds: Optional[dict] = None,
) -> RecordPrediction:
    """Run inference on a single spectrum tensor and return structured predictions.

    Args:
        x:          Spectrum intensity tensor, shape (1000,) or (1, 1000).
        model:      Trained MultitaskMLP or MultitaskMLPWithQRegression.
        dataset:    Dataset (or Subset) — used for schema attributes only, no records loaded.
        threshold:  Confidence cutoff for binary/categorical entries (ignored when
                    thresholds is provided).
        thresholds: Conformal thresholds from load_thresholds() or calibrate_model().
                    When provided, uses conformal prediction sets instead of threshold
                    cutoff; quantile intervals are conformal-adjusted.

    Returns:
        RecordPrediction with structural entries, element count ranges, and
        continuous feature ranges for generating extra ASP constraints.
    """
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    is_raw = isinstance(base_dataset, RawFeatureDataset)
    device = next(model.parameters()).device

    if x.dim() == 1:
        x = x.unsqueeze(0)
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(x)

    onehot_order: list[str] = base_dataset.onehot_feature_order if is_raw else []

    if thresholds is not None:
        num_quantiles = getattr(model, "num_quantiles", 1)
        conf_sets = sets_from_outputs(outputs, thresholds, 0, num_quantiles - 1)

        structural = _structural_from_conf_sets_single(
            conf_sets["binary"].squeeze(0),
            [c.squeeze(0) for c in conf_sets["categorical"]],
            onehot_order,
        )

        if is_raw:
            if conf_sets["quantile"] is not None:
                count_ranges, continuous_ranges = _process_conf_quantile_sets(
                    conf_sets["quantile"].squeeze(0),
                    base_dataset.continuous_feature_order,
                    base_dataset._cont_mean,
                    base_dataset._cont_std,
                )
            else:
                # Conformal quantile sets unavailable — fall back to raw quantile outputs.
                q_list = [q.squeeze(0) for q in outputs.get("quantile", [])]
                count_ranges, continuous_ranges = _process_quantile_outputs(
                    q_list,
                    base_dataset.continuous_feature_order,
                    base_dataset._cont_mean,
                    base_dataset._cont_std,
                )
        else:
            cat_logits = [c.squeeze(0) for c in outputs.get("categorical", [])]
            count_ranges = _count_ranges_from_categorical(cat_logits, onehot_order)
            continuous_ranges = {}
    else:
        cat_logits = [c.squeeze(0) for c in outputs.get("categorical", [])]
        structural = _structural_from_threshold(
            outputs["binary"].squeeze(0),
            cat_logits,
            onehot_order,
            threshold,
        )

        if is_raw:
            q_list = [q.squeeze(0) for q in outputs.get("quantile", [])]
            count_ranges, continuous_ranges = _process_quantile_outputs(
                q_list,
                base_dataset.continuous_feature_order,
                base_dataset._cont_mean,
                base_dataset._cont_std,
            )
        else:
            count_ranges = _count_ranges_from_categorical(cat_logits, onehot_order)
            continuous_ranges = {}

    return RecordPrediction(
        structural=structural,
        count_ranges=count_ranges,
        continuous_ranges=continuous_ranges,
    )


def run_inference(
    dataset: SpectrumDataset | Subset,
    model: torch.nn.Module,
    indices: Optional[Sequence[int]] = None,
    threshold: float = 0.5,
    thresholds: Optional[dict] = None,
    batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> list[RecordPrediction]:
    """Run model inference on dataset records and return per-record predictions.

    Args:
        dataset:    SpectrumDataset or Subset to run inference on.
        model:      Trained MultitaskMLP or MultitaskMLPWithQRegression.
        indices:    Record indices to run inference on; None runs all records.
        threshold:  Confidence cutoff for binary/categorical entries (ignored when
                    thresholds is provided).
        thresholds: Conformal thresholds from load_thresholds() or calibrate_model().
                    When provided, uses conformal prediction sets instead of threshold
                    cutoff for structural entries, and conformal-adjusted quantile
                    intervals for count/continuous ranges.
        batch_size: DataLoader batch size.
        device:     Inference device; inferred from model parameters if None.

    Returns:
        List of RecordPrediction in index order (or dataset order when indices is None).
    """
    if device is None:
        device = next(model.parameters()).device

    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    is_raw = isinstance(base_dataset, RawFeatureDataset)
    onehot_order: list[str] = base_dataset.onehot_feature_order if is_raw else []

    data = Subset(dataset, list(indices)) if indices is not None else dataset
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    num_quantiles = getattr(model, "num_quantiles", 1)
    results: list[RecordPrediction] = []
    model.eval()

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            outputs = model(x_batch)
            batch_n = x_batch.size(0)

            if thresholds is not None:
                conf_sets = sets_from_outputs(outputs, thresholds, 0, num_quantiles - 1)
                has_q_sets = conf_sets["quantile"] is not None

                for i in range(batch_n):
                    structural = _structural_from_conf_sets_single(
                        conf_sets["binary"][i],
                        [c[i] for c in conf_sets["categorical"]],
                        onehot_order,
                    )

                    if is_raw:
                        if has_q_sets:
                            count_ranges, continuous_ranges = _process_conf_quantile_sets(
                                conf_sets["quantile"][i],
                                base_dataset.continuous_feature_order,
                                base_dataset._cont_mean,
                                base_dataset._cont_std,
                            )
                        else:
                            q_list = [q[i] for q in outputs.get("quantile", [])]
                            count_ranges, continuous_ranges = _process_quantile_outputs(
                                q_list,
                                base_dataset.continuous_feature_order,
                                base_dataset._cont_mean,
                                base_dataset._cont_std,
                            )
                    else:
                        cat_list = [c[i] for c in outputs.get("categorical", [])]
                        count_ranges = _count_ranges_from_categorical(cat_list, onehot_order)
                        continuous_ranges = {}

                    results.append(RecordPrediction(structural, count_ranges, continuous_ranges))
            else:
                binary_logits = outputs["binary"]
                cat_logits_list = outputs.get("categorical", [])
                q_logits_list = outputs.get("quantile", [])

                for i in range(batch_n):
                    cat_i = [c[i] for c in cat_logits_list]
                    structural = _structural_from_threshold(
                        binary_logits[i], cat_i, onehot_order, threshold
                    )

                    if is_raw and q_logits_list:
                        count_ranges, continuous_ranges = _process_quantile_outputs(
                            [q[i] for q in q_logits_list],
                            base_dataset.continuous_feature_order,
                            base_dataset._cont_mean,
                            base_dataset._cont_std,
                        )
                    else:
                        count_ranges = _count_ranges_from_categorical(cat_i, onehot_order)
                        continuous_ranges = {}

                    results.append(RecordPrediction(structural, count_ranges, continuous_ranges))

    return results
