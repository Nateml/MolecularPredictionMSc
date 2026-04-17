from __future__ import annotations

import bisect
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# Bit-packing helpers (unchanged from original)
# ---------------------------------------------------------------------------

def unpack_bits_to_int_list(packed: bytes, n_bits: int) -> List[int]:
    out: List[int] = [0] * n_bits
    for i in range(n_bits):
        byte_val = packed[i // 8]
        bit = (byte_val >> (7 - (i % 8))) & 1
        out[i] = int(bit)
    return out


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def build_targets_from_schema(
    schema_path: str | Path,
) -> Tuple[
    int,                         # maccs_length
    List[str],                   # onehot_names  (ordered)
    List[str],                   # onehot_feature_order
    Dict[str, Tuple[int, int]],  # onehot_feature_to_indices  (into onehot block)
    List[str],                   # continuous_feature_order
]:
    """Parse a mixed schema file and return all structures needed by the dataset.

    Schema format
    -------------
    {
        "maccs_length": 167,
        "onehot_names": ["feat_A=x", "feat_A=y", "feat_B=p", ...],
        "feature_schema": {
            "feat_A": {
                "type": "onehot",
                "onehot_names": ["feat_A=x", "feat_A=y"]
            },
            "feat_B": {
                "type": "onehot",
                "onehot_names": ["feat_B=p", ...]
            },
            "mol_weight": {
                "type": "continuous"
            }
        }
    }

    Continuous features have **no** onehot_names; their raw float values are
    stored directly in each record under the key ``continuous_field`` (default
    ``"continuous_values"``).  The ordering of continuous features in that list
    must match the order they appear in ``feature_schema``.
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    maccs_length: int = int(payload.get("maccs_length", 167))
    onehot_names: List[str] = list(payload["onehot_names"])
    feature_schema: Dict[str, Dict] = payload["feature_schema"]

    # Map each one-hot name back to its parent feature
    onehot_to_feature: Dict[str, str] = {}
    for feat_name, spec in feature_schema.items():
        if spec.get("type", "onehot") == "onehot":
            for name in spec.get("onehot_names", []):
                onehot_to_feature[name] = feat_name

    # Build ordered feature list and per-feature index ranges inside the one-hot block
    onehot_feature_order: List[str] = []
    onehot_feature_to_indices: Dict[str, Tuple[int, int]] = {}
    idx = 0
    for name in onehot_names:
        feat = onehot_to_feature.get(name)
        if feat is None:
            raise ValueError(
                f"onehot_name '{name}' is not referenced by any feature in feature_schema"
            )
        if not onehot_feature_order or onehot_feature_order[-1] != feat:
            onehot_feature_order.append(feat)
            onehot_feature_to_indices[feat] = (idx, idx + 1)
        else:
            start, _ = onehot_feature_to_indices[feat]
            onehot_feature_to_indices[feat] = (start, idx + 1)
        idx += 1

    # Collect continuous features in schema declaration order
    continuous_feature_order: List[str] = [
        feat_name
        for feat_name, spec in feature_schema.items()
        if spec.get("type") == "continuous"
    ]

    return (
        maccs_length,
        onehot_names,
        onehot_feature_order,
        onehot_feature_to_indices,
        continuous_feature_order,
    )


def build_categorical_sizes_from_schema(schema_path: str | Path) -> Dict[str, int]:
    """Return {feature_name: n_classes} for every one-hot feature in the schema.

    Convenience utility for constructing model heads that match dataset ordering.
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {
        feat: len(spec["onehot_names"])
        for feat, spec in payload["feature_schema"].items()
        if spec.get("type", "onehot") == "onehot"
    }


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MixedFeatureDataset(Dataset):
    """Dataset with a *mixed* target vector: MACCS bits, one-hot categoricals,
    and raw continuous features.

    Record format (each element of the JSON list)
    ----------------------------------------------
    {
        "spectrum":              "<MoNA style string>",
        "combined_packed_b64":  "<base64>",   # MACCS bits + one-hot block
        "combined_nbits":       <int>,
        "continuous_values":    [1.234, 0.987, ...]   # one float per continuous feature
    }

    The ``continuous_values`` list must contain one entry per continuous feature
    declared in the schema, in the **same order** those features appear in
    ``feature_schema``.

    Returns
    -------
    (spectrum_tensor, targets_dict)

    spectrum_tensor : torch.FloatTensor  shape (spectrum_len,)

    targets_dict keys
        "binary"      : FloatTensor  shape (maccs_length,)   – MACCS bits
        "categorical" : List[LongTensor scalar]              – one per one-hot feature
        "continuous"  : FloatTensor  shape (n_continuous,)   – raw feature values
    """

    def __init__(
        self,
        data_path: str | Path,
        schema_path: str | Path,
        # ── field names ──────────────────────────────────────────────────────
        spectrum_field: str = "spectrum",
        combined_b64_field: str = "combined_packed_b64",
        combined_nbits_field: str = "combined_nbits",
        continuous_field: str = "continuous_values",
        # ── spectrum binning ─────────────────────────────────────────────────
        spectrum_len: int = 1000,
        mz_min: float = 0.0,
        mz_max: float = 1000.0,
        # ── continuous normalisation (optional) ──────────────────────────────
        continuous_mean: Optional[List[float]] = None,
        continuous_std: Optional[List[float]] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.schema_path = Path(schema_path)

        # Load records
        with open(self.data_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, list):
            raise ValueError("data_path must be a JSON file containing a list of records.")
        self.records = payload

        # Parse schema
        (
            self.maccs_length,
            self.onehot_names,
            self.onehot_feature_order,
            self.onehot_feature_to_indices,
            self.continuous_feature_order,
        ) = build_targets_from_schema(schema_path)

        # Field names
        self.spectrum_field = spectrum_field
        self.combined_b64_field = combined_b64_field
        self.combined_nbits_field = combined_nbits_field
        self.continuous_field = continuous_field

        # Spectrum binning
        self.spectrum_len = spectrum_len
        self.mz_min = mz_min
        self.mz_max = mz_max

        # Optional normalisation for continuous targets
        n_cont = len(self.continuous_feature_order)
        if continuous_mean is not None:
            if len(continuous_mean) != n_cont:
                raise ValueError(
                    f"continuous_mean length {len(continuous_mean)} != "
                    f"n_continuous {n_cont}"
                )
            self._cont_mean: Optional[torch.Tensor] = torch.tensor(
                continuous_mean, dtype=torch.float32
            )
        else:
            self._cont_mean = None

        if continuous_std is not None:
            if len(continuous_std) != n_cont:
                raise ValueError(
                    f"continuous_std length {len(continuous_std)} != "
                    f"n_continuous {n_cont}"
                )
            self._cont_std: Optional[torch.Tensor] = torch.tensor(
                continuous_std, dtype=torch.float32
            )
        else:
            self._cont_std = None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def n_continuous(self) -> int:
        return len(self.continuous_feature_order)

    @property
    def n_categorical(self) -> int:
        return len(self.onehot_feature_order)

    def categorical_sizes(self) -> Dict[str, int]:
        """Return {feature_name: n_classes} for all one-hot features."""
        return build_categorical_sizes_from_schema(self.schema_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spectrum_to_dense(self, spectrum: str) -> torch.Tensor:
        vec = torch.zeros(self.spectrum_len, dtype=torch.float32)
        if not spectrum:
            return vec
        scale = (self.spectrum_len - 1) / max(1e-12, self.mz_max - self.mz_min)
        for token in spectrum.strip().split():
            try:
                mz_str, intensity_str = token.split(":")
                mz = float(mz_str)
                intensity = float(intensity_str)
            except ValueError:
                continue
            if mz < self.mz_min or mz > self.mz_max:
                continue
            bin_idx = int(round((mz - self.mz_min) * scale))
            if intensity > vec[bin_idx]:
                vec[bin_idx] = intensity
        max_val = vec.max()
        if max_val > 0:
            vec = vec / max_val
        return vec

    def _decode_packed(self, record: dict, record_idx: int) -> Tuple[List[int], List[int]]:
        """Decode base64-packed bits → (maccs_bits, onehot_block)."""
        packed_b64 = record.get(self.combined_b64_field)
        n_bits = record.get(self.combined_nbits_field)
        if not isinstance(packed_b64, str) or not isinstance(n_bits, int):
            raise ValueError(
                f"Record {record_idx}: missing or invalid fields "
                f"'{self.combined_b64_field}' / '{self.combined_nbits_field}'"
            )
        packed = base64.b64decode(packed_b64)
        bits = unpack_bits_to_int_list(packed, n_bits)
        if len(bits) < self.maccs_length:
            raise ValueError(
                f"Record {record_idx}: packed vector length {len(bits)} "
                f"< maccs_length {self.maccs_length}"
            )
        return bits[: self.maccs_length], bits[self.maccs_length :]

    def _decode_continuous(self, record: dict, record_idx: int) -> torch.Tensor:
        """Read raw continuous values from the record and optionally normalise."""
        raw = record.get(self.continuous_field)
        if raw is None:
            if self.n_continuous == 0:
                return torch.zeros(0, dtype=torch.float32)
            raise ValueError(
                f"Record {record_idx}: missing field '{self.continuous_field}'"
            )
        if len(raw) != self.n_continuous:
            raise ValueError(
                f"Record {record_idx}: '{self.continuous_field}' has "
                f"{len(raw)} values but schema declares {self.n_continuous} "
                f"continuous features: {self.continuous_feature_order}"
            )
        t = torch.tensor(raw, dtype=torch.float32)
        if self._cont_mean is not None:
            t = t - self._cont_mean
        if self._cont_std is not None:
            t = t / self._cont_std.clamp(min=1e-8)
        return t

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]

        # ── Spectrum (input) ────────────────────────────────────────────
        x = self._spectrum_to_dense(str(record.get(self.spectrum_field, "")))

        # ── Binary MACCS targets ────────────────────────────────────────
        maccs_bits, onehot_block = self._decode_packed(record, idx)
        y_binary = torch.tensor(maccs_bits, dtype=torch.float32)

        # ── Categorical one-hot targets ─────────────────────────────────
        categorical_targets: List[torch.Tensor] = []
        for feat in self.onehot_feature_order:
            start, end = self.onehot_feature_to_indices[feat]
            slice_bits = onehot_block[start:end]
            class_idx = slice_bits.index(1) if any(slice_bits) else 0
            categorical_targets.append(torch.tensor(class_idx, dtype=torch.long))

        # ── Continuous targets ──────────────────────────────────────────
        y_continuous = self._decode_continuous(record, idx)

        return x, {
            "binary": y_binary,
            "categorical": categorical_targets,
            "continuous": y_continuous,
        }

class MixedRawFeatureDataset(Dataset):
    """Dataset that reads raw structural feature dicts and bins categorical
    features on-the-fly using the schema's bin edges.
 
    Expected record format
    ----------------------
    {
        "spectrum":            "42:18 51:7 ...",      # MoNA-style m/z:intensity
        "maccs":               [0, 0, 1, ...],        # 167 ints
        "structural_features": {                       # raw float values
            "exact_mass_da": 231.137,
            "aldehyde": 0,
            "alcohol": 0,
            ...
        }
    }
 
    Returns
    -------
    (spectrum_tensor, targets_dict)
 
    spectrum_tensor : FloatTensor  shape (spectrum_len,)
 
    targets_dict keys
        "binary"      : FloatTensor  shape (maccs_length,)  – MACCS bits
        "categorical" : List[LongTensor scalar]             – one per onehot feature,
                        class index derived from raw value + bin edges
        "continuous"  : FloatTensor  shape (n_continuous,)  – raw feature values,
                        optionally z-score normalised
    """
 
    def __init__(
        self,
        data_path: str | Path,
        schema_path: str | Path,
        # ── field names ──────────────────────────────────────────────────────
        spectrum_field: str = "spectrum",
        maccs_field: str = "maccs",
        features_field: str = "structural_features",
        # ── spectrum binning ─────────────────────────────────────────────────
        spectrum_len: int = 1000,
        mz_min: float = 0.0,
        mz_max: float = 1000.0,
        # ── continuous normalisation (optional) ──────────────────────────────
        continuous_mean: Optional[List[float]] = None,
        continuous_std: Optional[List[float]] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.schema_path = Path(schema_path)
 
        with open(self.data_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, list):
            raise ValueError("data_path must be a JSON file containing a list of records.")
        self.records = payload
 
        (
            self.maccs_length,
            self.onehot_feature_order,
            self.onehot_feature_edges,
            self.continuous_feature_order,
        ) = self.load_schema(schema_path)
 
        self.spectrum_field = spectrum_field
        self.maccs_field = maccs_field
        self.features_field = features_field
 
        self.spectrum_len = spectrum_len
        self.mz_min = mz_min
        self.mz_max = mz_max
 
        # Optional z-score normalisation for continuous targets
        n_cont = len(self.continuous_feature_order)
        if continuous_mean is not None:
            if len(continuous_mean) != n_cont:
                raise ValueError(
                    f"continuous_mean length {len(continuous_mean)} != n_continuous {n_cont}"
                )
            self._cont_mean: Optional[torch.Tensor] = torch.tensor(
                continuous_mean, dtype=torch.float32
            )
        else:
            self._cont_mean = None
 
        if continuous_std is not None:
            if len(continuous_std) != n_cont:
                raise ValueError(
                    f"continuous_std length {len(continuous_std)} != n_continuous {n_cont}"
                )
            self._cont_std: Optional[torch.Tensor] = torch.tensor(
                continuous_std, dtype=torch.float32
            )
        else:
            self._cont_std = None
 

    @classmethod
    def from_subset(cls, subset: Subset) -> "MixedRawFeatureDataset":
        """
        Create a new MixedRawFeatureDataset instance from a torch.utils.data.Subset of an existing MixedRawFeatureDataset.
        Normalisation is reset so it can be computed fresh on the new split.
        """
        original = subset.dataset
        # Build a new instance without calling __init__ (avoids re-reading files)
        new = cls.__new__(cls)
        new.__dict__ = {**original.__dict__}
        new.records = [original.records[i] for i in subset.indices]
        new._cont_mean = None
        new._cont_std = None
        return new

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
 
    @property
    def n_continuous(self) -> int:
        return len(self.continuous_feature_order)
 
    @property
    def n_categorical(self) -> int:
        return len(self.onehot_feature_order)
 
    def categorical_sizes(self) -> Dict[str, int]:
        """Return {feature_name: n_classes} for all onehot features."""
        return build_categorical_sizes_from_schema(self.schema_path)
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def normalise_continuous(self, mean: Optional[List[float]] = None, std: Optional[List[float]] = None) -> Tuple[List[float], List[float]]:
        """
        Compute mean and std from the continuous features in this dataset and store them for normalisation in __getitem__.
            If mean/std are provided, use those instead of computing from the data. This allows val/test datasets to be normalised using training stats.

        Returns the mean and std used for normalisation (either computed or provided).
        """

        if mean is not None or std is not None:
            assert mean is not None and std is not None, "Both mean and std must be provided together"
            self._cont_mean = torch.tensor(mean, dtype=torch.float32)
            self._cont_std = torch.tensor(std, dtype=torch.float32)
            return mean, std

        all_cont = torch.stack([self[i][1]["continuous"] for i in range(len(self))])
        # shape: (n_records, n_continuous)

        mean_t = all_cont.mean(dim=0)
        std_t  = all_cont.std(dim=0).clamp(min=1e-8)

        self._cont_mean = mean_t
        self._cont_std  = std_t

        return mean_t.tolist(), std_t.tolist()

    def set_normalisation(self, mean: List[float], std: List[float]) -> None:
        """Set the normalisation parameters for this dataset, to be applied in __getitem__.

        Should be called on val/test datasets after computing normalisation from the training dataset.
        """
        n_cont = len(self.continuous_feature_order)
        if len(mean) != n_cont or len(std) != n_cont:
            raise ValueError(
                f"Mean/std length {len(mean)} / {len(std)} does not match "
                f"n_continuous {n_cont} declared in schema"
            )
        self._cont_mean = torch.tensor(mean, dtype=torch.float32)
        self._cont_std  = torch.tensor(std, dtype=torch.float32)
 
    def _spectrum_to_dense(self, spectrum: str) -> torch.Tensor:
        vec = torch.zeros(self.spectrum_len, dtype=torch.float32)
        if not spectrum:
            return vec
        scale = (self.spectrum_len - 1) / max(1e-12, self.mz_max - self.mz_min)
        for token in spectrum.strip().split():
            try:
                mz_str, intensity_str = token.split(":")
                mz = float(mz_str)
                intensity = float(intensity_str)
            except ValueError:
                continue
            if mz < self.mz_min or mz > self.mz_max:
                continue
            bin_idx = int(round((mz - self.mz_min) * scale))
            if intensity > vec[bin_idx]:
                vec[bin_idx] = intensity
        max_val = vec.max()
        if max_val > 0:
            vec = vec / max_val
        return vec
    
    def load_schema(self, schema_path: str | Path):
        """Parse schema_mixed.json and return all structures needed by the dataset.
    
        Returns
        -------
        maccs_length            : int
        onehot_feature_order    : List[str]   – categorical features, schema order
        onehot_feature_edges    : Dict[str, List[float]]  – bin edges per feature
        continuous_feature_order: List[str]   – continuous features, schema order
        """
        with open(schema_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    
        maccs_length: int = int(payload.get("maccs_length", 167))
        feature_schema: Dict[str, Dict] = payload["feature_schema"]
    
        onehot_feature_order: List[str] = []
        onehot_feature_edges: Dict[str, List[float]] = {}
        continuous_feature_order: List[str] = []
    
        # Preserve schema declaration order (Python 3.7+ dicts are ordered)
        for feat_name, spec in feature_schema.items():
            if spec.get("type") == "onehot":
                onehot_feature_order.append(feat_name)
                onehot_feature_edges[feat_name] = spec["edges"]
            elif spec.get("type") == "continuous":
                continuous_feature_order.append(feat_name)
    
        return (
            maccs_length,
            onehot_feature_order,
            onehot_feature_edges,
            continuous_feature_order,
        )

    def _bin_value(self, value: float, edges: List[float]) -> int:
        """Convert a raw feature value to a bin class index using schema edges.
    
        The convention matches the schema labels:
            bin_0  →  value <= edges[0]
            bin_1  →  edges[0] < value <= edges[1]
            ...
            bin_N  →  value > edges[-1]
    
        bisect_left on the edges gives the correct index directly.
    
        Examples (edges = [0.0, 1.0, 4.0]):
            value=0   → bin 0  ("<= 0")
            value=0.5 → bin 1  ("> 0 and <= 1")
            value=4.0 → bin 2  ("> 1 and <= 4")
            value=5   → bin 3  ("> 4")
        """
        return bisect.bisect_left(edges, value)
 
    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
 
    def __len__(self) -> int:
        return len(self.records)
 
    def __getitem__(self, idx: int):
        record = self.records[idx]
 
        # ── Spectrum (model input) ──────────────────────────────────────
        x = self._spectrum_to_dense(str(record.get(self.spectrum_field, "")))
 
        # ── Binary MACCS target ─────────────────────────────────────────
        maccs_raw = record.get(self.maccs_field)
        if maccs_raw is None:
            raise ValueError(f"Record {idx}: missing field '{self.maccs_field}'")
        if len(maccs_raw) != self.maccs_length:
            raise ValueError(
                f"Record {idx}: '{self.maccs_field}' has {len(maccs_raw)} bits, "
                f"expected {self.maccs_length}"
            )
        y_binary = torch.tensor(maccs_raw, dtype=torch.float32)
 
        # ── Categorical + continuous targets ────────────────────────────
        features = record.get(self.features_field)
        if not isinstance(features, dict):
            raise ValueError(
                f"Record {idx}: '{self.features_field}' must be a dict of raw feature values"
            )
 
        # Categorical: bin each raw value using the schema's edges
        categorical_targets: List[torch.Tensor] = []
        for feat in self.onehot_feature_order:
            raw = features.get(feat)
            if raw is None:
                raise ValueError(
                    f"Record {idx}: '{self.features_field}' missing onehot feature '{feat}'"
                )
            class_idx = self._bin_value(float(raw), self.onehot_feature_edges[feat])
            categorical_targets.append(torch.tensor(class_idx, dtype=torch.long))
 
        # Continuous: read raw values directly
        cont_vals: List[float] = []
        for feat in self.continuous_feature_order:
            raw = features.get(feat)
            if raw is None:
                raise ValueError(
                    f"Record {idx}: '{self.features_field}' missing continuous feature '{feat}'"
                )
            cont_vals.append(float(raw))
 
        y_continuous = torch.tensor(cont_vals, dtype=torch.float32)
        if self._cont_mean is not None:
            y_continuous = y_continuous - self._cont_mean
        if self._cont_std is not None:
            y_continuous = y_continuous / self._cont_std.clamp(min=1e-8)
 
        return x, {
            "binary": y_binary,
            "categorical": categorical_targets,
            "continuous": y_continuous,
        }