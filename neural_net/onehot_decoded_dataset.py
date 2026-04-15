from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset


def unpack_bits_to_int_list(packed: bytes, n_bits: int) -> List[int]:
    out: List[int] = [0] * n_bits
    for i in range(n_bits):
        byte_val = packed[i // 8]
        bit = (byte_val >> (7 - (i % 8))) & 1
        out[i] = int(bit)
    return out


class OneHotDecodedDataset(Dataset):
    """Dataset that decodes packed one-hot feature vectors using a schema file.

    Expects records containing at least:
      - `spectrum` (MoNA style string)
      - `combined_packed_b64` (base64 packed bits)
      - `combined_nbits` (int)

    Also requires a schema JSON created by `one_hot_structural_features.py`.

    Returns items as `(spectrum_tensor, {"binary": tensor, "categorical": [tensor, ...]})`
    where `binary` is a float tensor of MACCS bits and `categorical` is a list
    of long tensors (class indices) in the same order as the schema-derived
    categorical features.
    """

    def __init__(
        self,
        data_path: str | Path,
        schema_path: str | Path,
        spectrum_field: str = "spectrum",
        combined_b64_field: str = "combined_packed_b64",
        combined_nbits_field: str = "combined_nbits",
        spectrum_len: int = 1000,
        mz_min: float = 0.0,
        mz_max: float = 1000.0,
    ) -> None:
        self.data_path = Path(data_path)
        self.schema_path = Path(schema_path)

        with open(self.data_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError("Input data must be a JSON list of records.")
        self.records = payload

        with open(self.schema_path, "r", encoding="utf-8") as f:
            schema_payload = json.load(f)

        # MACCS length (first section of feature vector)
        self.maccs_length: int = int(schema_payload.get("maccs_length", 167))

        # One-hot names list (in encoding order)
        self.onehot_names: List[str] = list(schema_payload["onehot_names"]) 

        # Feature schema: maps feature -> spec (contains onehot_names for that feature)
        self.feature_schema: Dict[str, Dict] = schema_payload["feature_schema"]

        # Build ordered feature list (preserves encoding order) and index ranges
        onehot_to_feature: Dict[str, str] = {}
        for feat_name, spec in self.feature_schema.items():
            for name in spec["onehot_names"]:
                onehot_to_feature[name] = feat_name

        # Derive feature order and start/end indices in the onehot block
        feature_order: List[str] = []
        feature_to_indices: Dict[str, Tuple[int, int]] = {}
        idx = 0
        for name in self.onehot_names:
            feat = onehot_to_feature.get(name)
            if feat is None:
                raise ValueError(f"Onehot name {name} not present in feature_schema")
            if not feature_order or feature_order[-1] != feat:
                feature_order.append(feat)
                feature_to_indices[feat] = (idx, idx + 1)
            else:
                start, end = feature_to_indices[feat]
                feature_to_indices[feat] = (start, end + 1)
            idx += 1

        self.feature_order = feature_order
        self.feature_to_indices = feature_to_indices

        # Field names
        self.spectrum_field = spectrum_field
        self.combined_b64_field = combined_b64_field
        self.combined_nbits_field = combined_nbits_field

        self.spectrum_len = spectrum_len
        self.mz_min = mz_min
        self.mz_max = mz_max

    def __len__(self) -> int:
        return len(self.records)

    def _spectrum_to_dense(self, spectrum: str) -> torch.Tensor:
        vec = torch.zeros(self.spectrum_len, dtype=torch.float32)
        if not spectrum:
            return vec
        scale = (self.spectrum_len - 1) / max(1e-12, (self.mz_max - self.mz_min))
        for token in spectrum.strip().split():
            try:
                mz_str, intensity_str = token.split(":")
                mz = float(mz_str)
                intensity = float(intensity_str)
            except ValueError:
                continue
            if mz < self.mz_min or mz > self.mz_max:
                continue
            idx = int(round((mz - self.mz_min) * scale))
            if intensity > vec[idx]:
                vec[idx] = intensity
        max_val = torch.max(vec)
        if max_val > 0:
            vec = vec / max_val
        return vec

    def __getitem__(self, idx: int):
        record = self.records[idx]

        spectrum_str = str(record.get(self.spectrum_field, ""))
        x = self._spectrum_to_dense(spectrum_str)

        packed_b64 = record.get(self.combined_b64_field)
        n_bits = record.get(self.combined_nbits_field)
        if not isinstance(packed_b64, str) or not isinstance(n_bits, int):
            raise ValueError(f"Record {idx} missing packed target fields: {self.combined_b64_field}, {self.combined_nbits_field}")

        packed = base64.b64decode(packed_b64)
        bits = unpack_bits_to_int_list(packed, n_bits)

        if len(bits) < self.maccs_length:
            raise ValueError(f"Packed vector too short: {len(bits)} < {self.maccs_length}")

        # Binary MACCS head
        binary_bits = bits[: self.maccs_length]
        y_binary = torch.tensor(binary_bits, dtype=torch.float32)

        # One-hot block
        onehot_block = bits[self.maccs_length :]

        categorical_targets: List[torch.Tensor] = []
        for feat in self.feature_order:
            start, end = self.feature_to_indices[feat]
            slice_bits = onehot_block[start:end]
            # if no bit set, fallback to 0
            if not any(slice_bits):
                class_idx = 0
            else:
                # prefer single-hot: index of first 1
                class_idx = int(slice_bits.index(1))
            categorical_targets.append(torch.tensor(class_idx, dtype=torch.long))

        return x, {"binary": y_binary, "categorical": categorical_targets}


def build_categorical_sizes_from_schema(schema_path: str | Path) -> Dict[str, int]:
    """Utility: build categorical_sizes dict (feature -> n_classes) from schema file.

    Use this dict when constructing `MultitaskMLP(categorical_sizes=...)` so the
    model heads match dataset ordering.
    """
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_payload = json.load(f)
    feature_schema = schema_payload["feature_schema"]
    return {feat: len(spec["onehot_names"]) for feat, spec in feature_schema.items()}
