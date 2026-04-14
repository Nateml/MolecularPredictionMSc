from __future__ import annotations

import base64
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def unpack_bits_to_float_tensor(packed: bytes, n_bits: int) -> torch.Tensor:
    """Unpack MSB-first packed bits to float tensor of shape (n_bits,)."""
    out = torch.zeros(n_bits, dtype=torch.float32)
    for i in range(n_bits):
        byte_val = packed[i // 8]
        bit = (byte_val >> (7 - (i % 8))) & 1
        out[i] = float(bit)
    return out


def spectrum_string_to_dense(
    spectrum: str,
    spectrum_len: int = 1000,
    mz_min: float = 0.0,
    mz_max: float = 1000.0,
) -> torch.Tensor:
    """Convert a MoNA-style 'mz:intensity ...' string to fixed-length dense vector."""
    vec = torch.zeros(spectrum_len, dtype=torch.float32)
    if not spectrum:
        return vec

    scale = (spectrum_len - 1) / max(1e-12, (mz_max - mz_min))
    for token in spectrum.strip().split():
        try:
            mz_str, intensity_str = token.split(":")
            mz = float(mz_str)
            intensity = float(intensity_str)
        except ValueError:
            continue

        if mz < mz_min or mz > mz_max:
            continue
        idx = int(round((mz - mz_min) * scale))
        if intensity > vec[idx]:
            vec[idx] = intensity

    max_val = torch.max(vec)
    if max_val > 0:
        vec = vec / max_val
    return vec


class PackedMoNADataset(Dataset):
    """Dataset for packed bit targets exported by one_hot_structural_features.py.

    Each item returns:
      spectra_tensor, {"binary": target_tensor}

    This matches the expected format in train.py.
    """

    def __init__(
        self,
        data_path: str | Path,
        target_field_b64: str = "combined_packed_b64",
        target_field_nbits: str = "combined_nbits",
        spectrum_field: str = "spectrum",
        spectrum_len: int = 1000,
        mz_min: float = 0.0,
        mz_max: float = 1000.0,
    ) -> None:
        self.data_path = Path(data_path)
        with open(self.data_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        if not isinstance(payload, list):
            raise ValueError("Input data must be a JSON list of records.")

        self.records = payload
        self.target_field_b64 = target_field_b64
        self.target_field_nbits = target_field_nbits
        self.spectrum_field = spectrum_field
        self.spectrum_len = spectrum_len
        self.mz_min = mz_min
        self.mz_max = mz_max

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]

        spectrum_str = str(record.get(self.spectrum_field, ""))
        x = spectrum_string_to_dense(
            spectrum=spectrum_str,
            spectrum_len=self.spectrum_len,
            mz_min=self.mz_min,
            mz_max=self.mz_max,
        )

        packed_b64 = record.get(self.target_field_b64)
        n_bits = record.get(self.target_field_nbits)
        if not isinstance(packed_b64, str) or not isinstance(n_bits, int):
            raise ValueError(
                f"Record {idx} missing packed target fields: "
                f"{self.target_field_b64}, {self.target_field_nbits}"
            )

        packed = base64.b64decode(packed_b64)
        y = unpack_bits_to_float_tensor(packed, n_bits)

        return x, {"binary": y}
