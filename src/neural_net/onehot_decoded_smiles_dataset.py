from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


def unpack_bits_to_int_list(packed: bytes, n_bits: int) -> List[int]:
    out: List[int] = [0] * n_bits
    for i in range(n_bits):
        byte_val = packed[i // 8]
        bit = (byte_val >> (7 - (i % 8))) & 1
        out[i] = int(bit)
    return out


class OneHotDecodedSmilesDataset(Dataset):
    """Dataset that decodes packed one-hot feature vectors using a schema file.

    Expects records containing at least:
      - `smiles` (or a custom field name)
      - `combined_packed_b64` (base64 packed bits)
      - `combined_nbits` (int)

    Also requires a schema JSON created by `one_hot_structural_features.py`.

    Returns items as `(smiles, {"binary": tensor, "categorical": [tensor, ...]})`
    where `binary` is a float tensor of MACCS bits and `categorical` is a list
    of long tensors (class indices) in the same order as the schema-derived
    categorical features.
    """

    def __init__(
        self,
        data_path: str | Path,
        schema_path: str | Path,
        smiles_field: str = "smiles",
        combined_b64_field: str = "combined_packed_b64",
        combined_nbits_field: str = "combined_nbits",
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

        self.maccs_length: int = int(schema_payload.get("maccs_length", 167))
        self.onehot_names: List[str] = list(schema_payload["onehot_names"])
        self.feature_schema: Dict[str, Dict] = schema_payload["feature_schema"]

        onehot_to_feature: Dict[str, str] = {}
        for feat_name, spec in self.feature_schema.items():
            for name in spec["onehot_names"]:
                onehot_to_feature[name] = feat_name

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

        self.smiles_field = smiles_field
        self.combined_b64_field = combined_b64_field
        self.combined_nbits_field = combined_nbits_field

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]

        smiles = str(record.get(self.smiles_field, "")).strip()

        packed_b64 = record.get(self.combined_b64_field)
        n_bits = record.get(self.combined_nbits_field)
        if not isinstance(packed_b64, str) or not isinstance(n_bits, int):
            raise ValueError(
                f"Record {idx} missing packed target fields: "
                f"{self.combined_b64_field}, {self.combined_nbits_field}"
            )

        packed = base64.b64decode(packed_b64)
        bits = unpack_bits_to_int_list(packed, n_bits)

        if len(bits) < self.maccs_length:
            raise ValueError(f"Packed vector too short: {len(bits)} < {self.maccs_length}")

        binary_bits = bits[: self.maccs_length]
        y_binary = torch.tensor(binary_bits, dtype=torch.float32)

        onehot_block = bits[self.maccs_length :]

        categorical_targets: List[torch.Tensor] = []
        for feat in self.feature_order:
            start, end = self.feature_to_indices[feat]
            slice_bits = onehot_block[start:end]
            if not any(slice_bits):
                class_idx = 0
            else:
                class_idx = int(slice_bits.index(1))
            categorical_targets.append(torch.tensor(class_idx, dtype=torch.long))

        return smiles, {"binary": y_binary, "categorical": categorical_targets}


def build_categorical_sizes_from_schema(schema_path: str | Path) -> Dict[str, int]:
    """Utility: build categorical_sizes dict (feature -> n_classes) from schema file."""
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_payload = json.load(f)
    feature_schema = schema_payload["feature_schema"]
    return {feat: len(spec["onehot_names"]) for feat, spec in feature_schema.items()}
