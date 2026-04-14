from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


def parse_spectrum_to_vector(spectrum: str, spectrum_len: int = 1000) -> torch.Tensor:
    """Convert 'mz:intensity ...' text into a fixed-length intensity vector.

    Each peak is assigned to the nearest integer m/z bin in [0, spectrum_len-1].
    Intensities are max-normalized to [0, 1] per spectrum.
    """
    x = torch.zeros(spectrum_len, dtype=torch.float32)
    if not spectrum:
        return x

    for token in spectrum.strip().split():
        try:
            mz_str, inten_str = token.split(":", 1)
            mz = int(round(float(mz_str)))
            intensity = float(inten_str)
        except (ValueError, TypeError):
            continue

        if 0 <= mz < spectrum_len:
            x[mz] += intensity

    max_val = torch.max(x)
    if max_val > 0:
        x = x / max_val
    return x


class SpectrumJsonDataset(Dataset):
    """Dataset for JSON records containing spectrum text + vector targets.

    Expected record shape (minimum):
        {
            "spectrum": "43.02:100 45.03:12 ...",
            "feature_vector_onehot": [0, 1, 0, ...]
        }

    Returned sample:
        spectrum_tensor, {"binary": target_tensor}
    """

    def __init__(
        self,
        json_path: str | Path,
        target_field: str = "feature_vector_onehot",
        spectrum_len: int = 1000,
        target_start: int = 0,
        target_end: int | None = None,
    ):
        self.json_path = Path(json_path)
        self.target_field = target_field
        self.spectrum_len = spectrum_len
        self.target_start = target_start
        self.target_end = target_end

        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Input JSON must be a list of records.")

        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        dropped = 0

        for rec in data:
            spectrum = rec.get("spectrum")
            target = rec.get(self.target_field)

            if not isinstance(spectrum, str) or not isinstance(target, list):
                dropped += 1
                continue

            try:
                target_tensor = torch.tensor([float(v) for v in target], dtype=torch.float32)
            except (TypeError, ValueError):
                dropped += 1
                continue

            if self.target_end is not None:
                target_tensor = target_tensor[self.target_start:self.target_end]
            else:
                target_tensor = target_tensor[self.target_start:]

            x = parse_spectrum_to_vector(spectrum, spectrum_len=self.spectrum_len)
            self.samples.append((x, target_tensor))

        if not self.samples:
            raise ValueError("No valid records found for dataset construction.")

        self.num_binary = int(self.samples[0][1].shape[0])
        self.dropped = dropped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y = self.samples[idx]
        return x, {"binary": y}


def compute_pos_weight(dataset: SpectrumJsonDataset) -> torch.Tensor:
    """Compute BCE pos_weight for multilabel imbalance handling."""
    y = torch.stack([sample[1] for sample in dataset.samples], dim=0)
    pos = y.sum(dim=0)
    neg = y.shape[0] - pos
    # Avoid divide-by-zero for always-positive labels.
    pos_weight = neg / torch.clamp(pos, min=1.0)
    return pos_weight
