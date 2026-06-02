from __future__ import annotations

import base64
import bisect
import json
import pickle
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Subset


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------

def unpack_bits_to_int_list(packed: bytes, n_bits: int) -> List[int]:
    out: List[int] = [0] * n_bits
    for i in range(n_bits):
        byte_val = packed[i // 8]
        out[i] = (byte_val >> (7 - (i % 8))) & 1
    return out


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------

def build_targets_from_schema(
    schema_path: str | Path,
) -> Tuple[
    int,                         # maccs_length
    List[str],                   # onehot_names  (ordered)
    List[str],                   # onehot_feature_order
    Dict[str, Tuple[int, int]],  # onehot_feature_to_indices (into onehot block)
    List[str],                   # continuous_feature_order
]:
    """Parse a packed-bit schema file.

    Schema format
    -------------
    {
        "maccs_length": 167,
        "onehot_names": ["feat_A=x", "feat_A=y", "feat_B=p", ...],
        "feature_schema": {
            "feat_A": {"type": "onehot", "onehot_names": ["feat_A=x", "feat_A=y"]},
            "feat_B": {"type": "onehot", "onehot_names": ["feat_B=p", ...]},
            "mol_weight": {"type": "continuous"}
        }
    }
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    maccs_length: int = int(payload.get("maccs_length", 167))
    onehot_names: List[str] = list(payload.get("onehot_names", []))
    feature_schema: Dict[str, Dict] = payload.get("feature_schema", {})

    onehot_to_feature: Dict[str, str] = {}
    for feat_name, spec in feature_schema.items():
        if spec.get("type", "onehot") == "onehot":
            for name in spec.get("onehot_names", []):
                onehot_to_feature[name] = feat_name

    onehot_feature_order: List[str] = []
    onehot_feature_to_indices: Dict[str, Tuple[int, int]] = {}
    for idx, name in enumerate(onehot_names):
        feat = onehot_to_feature.get(name)
        if feat is None:
            raise ValueError(
                f"onehot_name '{name}' not referenced by any feature in feature_schema"
            )
        if not onehot_feature_order or onehot_feature_order[-1] != feat:
            onehot_feature_order.append(feat)
            onehot_feature_to_indices[feat] = (idx, idx + 1)
        else:
            start, _ = onehot_feature_to_indices[feat]
            onehot_feature_to_indices[feat] = (start, idx + 1)

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


def build_raw_targets_from_schema(
    schema_path: str | Path,
) -> Tuple[
    int,                          # maccs_length
    List[str],                    # onehot_feature_order
    Dict[str, List[float]],       # onehot_feature_edges (bin boundaries per feature)
    List[str],                    # continuous_feature_order
]:
    """Parse a raw-feature schema file (uses bin edges instead of onehot_names).

    Schema format
    -------------
    {
        "maccs_length": 167,
        "feature_schema": {
            "alcohol":    {"type": "onehot", "edges": [0.0, 1.0, 4.0]},
            "mol_weight": {"type": "continuous"}
        }
    }
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    maccs_length: int = int(payload.get("maccs_length", 167))
    feature_schema: Dict[str, Dict] = payload["feature_schema"]

    onehot_feature_order: List[str] = []
    onehot_feature_edges: Dict[str, List[float]] = {}
    continuous_feature_order: List[str] = []

    for feat_name, spec in feature_schema.items():
        if spec.get("type") == "onehot":
            onehot_feature_order.append(feat_name)
            onehot_feature_edges[feat_name] = spec["edges"]
        elif spec.get("type") == "continuous":
            continuous_feature_order.append(feat_name)

    return maccs_length, onehot_feature_order, onehot_feature_edges, continuous_feature_order


def build_categorical_sizes_from_schema(schema_path: str | Path) -> Dict[str, int]:
    """Build {feature_name: n_classes} for all one-hot features."""
    with open(schema_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        feat: len(spec["onehot_names"])
        for feat, spec in payload["feature_schema"].items()
        if spec.get("type", "onehot") == "onehot"
    }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SpectrumDataset(Dataset, ABC):
    """Abstract base for EI-MS spectrum → molecular feature datasets.

    Shared responsibilities
    -----------------------
    - Byte-offset streaming index over the JSON array (build once, cache to .idx)
    - Schema parsing via _parse_schema() (overridable by subclasses)
    - Spectrum string → dense intensity tensor
    - Base64-packed bit decoding (MACCS + one-hot block)
    - Categorical class-index extraction from the one-hot block
    - Lazy ``records`` property for split_dataset compatibility
    - from_subset() classmethod
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
        self.spectrum_field = spectrum_field
        self.combined_b64_field = combined_b64_field
        self.combined_nbits_field = combined_nbits_field
        self.spectrum_len = spectrum_len
        self.mz_min = mz_min
        self.mz_max = mz_max
        self._index: List[Tuple[int, int]] = self._build_or_load_index()
        self._parse_schema()

    def _parse_schema(self) -> None:
        """Populate schema-derived attributes. Override for non-packed-bit schemas."""
        (
            self.maccs_length,
            self.onehot_names,
            self.onehot_feature_order,
            self.onehot_feature_to_indices,
            self.continuous_feature_order,
        ) = build_targets_from_schema(self.schema_path)

    # ------------------------------------------------------------------
    # Streaming index
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return self.data_path.with_name(self.data_path.name + ".idx")

    def _build_or_load_index(self) -> List[Tuple[int, int]]:
        idx_path = self._index_path()
        if idx_path.exists():
            with open(idx_path, "r", encoding="utf-8") as fh:
                return [tuple(pair) for pair in json.load(fh)]

        index: List[Tuple[int, int]] = []
        bufsize = 1 << 20
        with open(self.data_path, "rb") as fh:
            pos = 0
            while True:
                b = fh.read(1)
                if not b:
                    raise ValueError("Unexpected end of file while locating JSON array start")
                pos += 1
                if b == b"[":
                    break

            depth = 0
            in_string = False
            escape = False
            obj_start = None
            while True:
                chunk = fh.read(bufsize)
                if not chunk:
                    break
                for i, byte in enumerate(chunk):
                    b = bytes([byte])
                    if in_string:
                        if escape:
                            escape = False
                        elif b == b"\\":
                            escape = True
                        elif b == b'"':
                            in_string = False
                    else:
                        if b == b'"':
                            in_string = True
                        elif b == b"{":
                            if depth == 0:
                                obj_start = pos + i
                            depth += 1
                        elif b == b"}":
                            depth -= 1
                            if depth == 0 and obj_start is not None:
                                index.append((obj_start, pos + i + 1))
                                obj_start = None
                        elif b == b"]" and depth == 0:
                            break
                else:
                    pos += len(chunk)
                    continue
                break

        # If we found no objects using array parsing, try newline-delimited JSON (NDJSON)
        # Some export formats store one JSON object per line rather than a top-level array.
        if not index:
            try:
                index_nd: List[Tuple[int, int]] = []
                with open(self.data_path, "rb") as fh2:
                    while True:
                        start = fh2.tell()
                        line = fh2.readline()
                        if not line:
                            break
                        stripped = line.lstrip()
                        if not stripped:
                            continue
                        if stripped.startswith(b"{"):
                            # validate JSON quickly
                            try:
                                json.loads(line.decode("utf-8"))
                            except Exception:
                                # not a valid JSON object line; skip
                                continue
                            end = fh2.tell()
                            index_nd.append((start, end))

                if index_nd:
                    index = index_nd
            except Exception:
                # If fallback fails, keep index as-is (possibly empty)
                pass

        with open(idx_path, "w", encoding="utf-8") as fh:
            json.dump(index, fh)
        return index

    def _load_record(self, idx: int) -> dict:
        cache = self.__dict__.get("_records_cache")
        if cache is not None:
            return cache[idx]
        start, end = self._index[idx]
        with open(self.data_path, "rb") as fh:
            fh.seek(start)
            return json.loads(fh.read(end - start).decode("utf-8"))

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_records_cache", None)
        return state

    @property
    def records(self) -> List[dict]:
        """All records as a list of dicts. Loaded lazily and cached.

        Exists for compatibility with split_dataset. Avoid on very large datasets.
        """
        if "_records_cache" not in self.__dict__:
            with ThreadPoolExecutor() as pool:
                self._records_cache = list(pool.map(self._load_record_from_file, range(len(self._index))))
        return self._records_cache

    def _load_record_from_file(self, idx: int) -> dict:
        start, end = self._index[idx]
        with open(self.data_path, "rb") as fh:
            fh.seek(start)
            return json.loads(fh.read(end - start).decode("utf-8"))

    def iter_smiles(self) -> List[str]:
        """Return SMILES string for each record without loading full records into memory.

        Uses a single file open with sequential seeks — much cheaper than loading
        all records for split grouping. Returns empty string when SMILES is absent.
        """
        result = []
        with open(self.data_path, "rb") as fh:
            for i, (start, end) in enumerate(self._index):
                fh.seek(start)
                record = json.loads(fh.read(end - start).decode("utf-8"))
                smiles = record.get("smiles")
                if smiles is None:
                    compound = record.get("compound")
                    if isinstance(compound, dict):
                        smiles = compound.get("smiles")
                result.append(str(smiles).strip() if smiles else f"__record_{i}")
        return result

    # ------------------------------------------------------------------
    # Subset construction
    # ------------------------------------------------------------------

    @classmethod
    def from_subset(cls, subset: Subset) -> "SpectrumDataset":
        """Create a new dataset containing only the indices in subset.

        Avoids re-reading or re-indexing the file. Clears the records cache
        so it reflects only the subset's records.
        """
        original = subset.dataset
        new = cls.__new__(cls)
        new.__dict__ = {**original.__dict__}
        new._index = [original._index[i] for i in subset.indices]
        new.__dict__.pop("_records_cache", None)
        return new

    # ------------------------------------------------------------------
    # Properties / utilities
    # ------------------------------------------------------------------

    @property
    def n_categorical(self) -> int:
        return len(self.onehot_feature_order)

    def categorical_sizes(self) -> Dict[str, int]:
        return build_categorical_sizes_from_schema(self.schema_path)

    # ------------------------------------------------------------------
    # Shared decode helpers
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
        bits = unpack_bits_to_int_list(base64.b64decode(packed_b64), n_bits)
        if len(bits) < self.maccs_length:
            raise ValueError(
                f"Record {record_idx}: packed vector length {len(bits)} "
                f"< maccs_length {self.maccs_length}"
            )
        return bits[: self.maccs_length], bits[self.maccs_length :]

    def _decode_categorical(self, onehot_block: List[int]) -> List[torch.Tensor]:
        """Extract one class index per one-hot feature from the onehot block."""
        targets: List[torch.Tensor] = []
        for feat in self.onehot_feature_order:
            start, end = self.onehot_feature_to_indices[feat]
            sl = onehot_block[start:end]
            class_idx = sl.index(1) if any(sl) else 0
            targets.append(torch.tensor(class_idx, dtype=torch.long))
        return targets

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int): ...


# ---------------------------------------------------------------------------
# Concrete datasets
# ---------------------------------------------------------------------------

class PackedDataset(SpectrumDataset):
    """Targets: binary + categorical.

    Expects records with MACCS bits and categorical features pre-binned and
    stored as base64-packed bits (``combined_packed_b64`` / ``combined_nbits``).
    Use when all continuous features have been discretised offline.
    """

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        record = self._load_record(idx)
        x = self._spectrum_to_dense(str(record.get(self.spectrum_field, "")))
        maccs_bits, onehot_block = self._decode_packed(record, idx)
        return x, {
            "binary": torch.tensor(maccs_bits, dtype=torch.float32),
            "categorical": self._decode_categorical(onehot_block),
        }


class PackedContinuousDataset(SpectrumDataset):
    """Targets: binary + categorical + continuous.

    Like PackedDataset but also reads raw continuous feature values from
    ``continuous_values`` in each record. Supports optional z-score normalisation.
    Use when categoricals are pre-binned (packed bits) but continuous features
    are kept as raw floats.
    """

    def __init__(
        self,
        data_path: str | Path,
        schema_path: str | Path,
        spectrum_field: str = "spectrum",
        combined_b64_field: str = "combined_packed_b64",
        combined_nbits_field: str = "combined_nbits",
        continuous_field: str = "continuous_values",
        spectrum_len: int = 1000,
        mz_min: float = 0.0,
        mz_max: float = 1000.0,
        continuous_mean: Optional[List[float]] = None,
        continuous_std: Optional[List[float]] = None,
    ) -> None:
        super().__init__(
            data_path, schema_path,
            spectrum_field, combined_b64_field, combined_nbits_field,
            spectrum_len, mz_min, mz_max,
        )
        self.continuous_field = continuous_field
        self._cont_mean, self._cont_std = self._make_norm_tensors(
            continuous_mean, continuous_std, len(self.continuous_feature_order)
        )

    @classmethod
    def from_subset(cls, subset: Subset) -> "PackedContinuousDataset":
        new = super().from_subset(subset)
        new._cont_mean = None
        new._cont_std = None
        return new

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @property
    def n_continuous(self) -> int:
        return len(self.continuous_feature_order)

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        record = self._load_record(idx)
        x = self._spectrum_to_dense(str(record.get(self.spectrum_field, "")))
        maccs_bits, onehot_block = self._decode_packed(record, idx)
        return x, {
            "binary": torch.tensor(maccs_bits, dtype=torch.float32),
            "categorical": self._decode_categorical(onehot_block),
            "continuous": self._decode_continuous(record, idx),
        }

    def _decode_continuous(self, record: dict, record_idx: int) -> torch.Tensor:
        raw = record.get(self.continuous_field)
        if raw is None:
            if self.n_continuous == 0:
                return torch.zeros(0, dtype=torch.float32)
            raise ValueError(
                f"Record {record_idx}: missing field '{self.continuous_field}'"
            )
        if len(raw) != self.n_continuous:
            raise ValueError(
                f"Record {record_idx}: '{self.continuous_field}' has {len(raw)} values "
                f"but schema declares {self.n_continuous} continuous features: "
                f"{self.continuous_feature_order}"
            )
        t = torch.tensor(raw, dtype=torch.float32)
        if self._cont_mean is not None:
            t = t - self._cont_mean
        if self._cont_std is not None:
            t = t / self._cont_std.clamp(min=1e-8)
        return t

    @staticmethod
    def _make_norm_tensors(
        mean: Optional[List[float]],
        std: Optional[List[float]],
        n: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if mean is not None and len(mean) != n:
            raise ValueError(f"continuous_mean length {len(mean)} != n_continuous {n}")
        if std is not None and len(std) != n:
            raise ValueError(f"continuous_std length {len(std)} != n_continuous {n}")
        mean_t = torch.tensor(mean, dtype=torch.float32) if mean is not None else None
        std_t = torch.tensor(std, dtype=torch.float32) if std is not None else None
        return mean_t, std_t


class RawFeatureDataset(SpectrumDataset):
    """Targets: binary + categorical + continuous.

    Expects records with MACCS bits as a plain int list and structural features
    as a dict of raw float values. Categorical features are binned on-the-fly
    at __getitem__ time using bin edges from the schema. Supports z-score
    normalisation for continuous features, computed lazily via normalise_continuous().

    Use when feature values are stored raw and binning decisions are deferred
    to load time.

    Record format
    -------------
    {
        "spectrum":            "42:18 51:7 ...",
        "maccs":               [0, 0, 1, ...],       # maccs_length ints
        "structural_features": {                       # raw float values
            "alcohol":    1,
            "mol_weight": 231.137,
            ...
        }
    }
    """

    def __init__(
        self,
        data_path: str | Path,
        schema_path: str | Path,
        spectrum_field: str = "spectrum",
        maccs_field: str = "maccs",
        features_field: str = "structural_features",
        spectrum_len: int = 1000,
        mz_min: float = 0.0,
        mz_max: float = 1000.0,
        continuous_mean: Optional[List[float]] = None,
        continuous_std: Optional[List[float]] = None,
    ) -> None:
        # Pass dummy values for the packed-bit field names — not used by this dataset
        super().__init__(
            data_path, schema_path,
            spectrum_field, "", "",
            spectrum_len, mz_min, mz_max,
        )
        self.maccs_field = maccs_field
        self.features_field = features_field
        self._cont_mean, self._cont_std = PackedContinuousDataset._make_norm_tensors(
            continuous_mean, continuous_std, len(self.continuous_feature_order)
        )

    def _parse_schema(self) -> None:
        (
            self.maccs_length,
            self.onehot_feature_order,
            self.onehot_feature_edges,
            self.continuous_feature_order,
        ) = build_raw_targets_from_schema(self.schema_path)
        # Not used by this dataset but expected by base properties
        self.onehot_names: List[str] = []
        self.onehot_feature_to_indices: Dict[str, Tuple[int, int]] = {}

    @classmethod
    def from_subset(cls, subset: Subset) -> "RawFeatureDataset":
        new = super().from_subset(subset)
        new._cont_mean = None
        new._cont_std = None
        return new

    def save(self, path: Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @property
    def n_continuous(self) -> int:
        return len(self.continuous_feature_order)

    def normalise_continuous(
        self,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
    ) -> Tuple[List[float], List[float]]:
        """Compute or assign z-score normalisation stats for continuous targets.

        If mean/std provided, use them directly (e.g. to apply training stats to
        val/test splits). Otherwise scan the dataset to compute them.
        Returns (mean, std) as lists.
        """
        if mean is not None or std is not None:
            assert mean is not None and std is not None, \
                "Both mean and std must be provided together"
            self._cont_mean = torch.tensor(mean, dtype=torch.float32)
            self._cont_std = torch.tensor(std, dtype=torch.float32)
            return mean, std

        pin_memory = torch.cuda.is_available()
        loader = torch.utils.data.DataLoader(
            self, batch_size=512, shuffle=False, num_workers=4, pin_memory=pin_memory
        )
        n = 0
        running_sum = running_sum_sq = None
        for batch in loader:
            cont = batch[1]["continuous"].float()
            if running_sum is None:
                running_sum = cont.sum(dim=0)
                running_sum_sq = (cont ** 2).sum(dim=0)
            else:
                running_sum += cont.sum(dim=0)
                running_sum_sq += (cont ** 2).sum(dim=0)
            n += cont.size(0)

        mean_t = running_sum / n
        std_t = (running_sum_sq / n - mean_t ** 2).sqrt().clamp(min=1e-8)
        self._cont_mean = mean_t
        self._cont_std = std_t
        return mean_t.tolist(), std_t.tolist()

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        record = self._load_record(idx)
        x = self._spectrum_to_dense(str(record.get(self.spectrum_field, "")))

        maccs_raw = record.get(self.maccs_field)
        if maccs_raw is None:
            raise ValueError(f"Record {idx}: missing field '{self.maccs_field}'")
        if len(maccs_raw) != self.maccs_length:
            raise ValueError(
                f"Record {idx}: '{self.maccs_field}' has {len(maccs_raw)} bits, "
                f"expected {self.maccs_length}"
            )
        y_binary = torch.tensor(maccs_raw, dtype=torch.float32)

        features = record.get(self.features_field)
        if not isinstance(features, dict):
            raise ValueError(
                f"Record {idx}: '{self.features_field}' must be a dict of raw feature values"
            )

        categorical_targets: List[torch.Tensor] = []
        for feat in self.onehot_feature_order:
            raw = features.get(feat)
            if raw is None:
                raise ValueError(
                    f"Record {idx}: '{self.features_field}' missing feature '{feat}'"
                )
            class_idx = bisect.bisect_left(self.onehot_feature_edges[feat], float(raw))
            categorical_targets.append(torch.tensor(class_idx, dtype=torch.long))

        cont_vals: List[float] = []
        for feat in self.continuous_feature_order:
            raw = features.get(feat)
            if raw is None:
                raise ValueError(
                    f"Record {idx}: '{self.features_field}' missing feature '{feat}'"
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def _detect_schema_type(schema_path: str | Path) -> str:
    """Inspect schema to determine which dataset class to use.

    Returns one of: "packed", "raw".
    """
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)

    if "continuous_feature_names" in schema:
        return "raw"

    return "packed"


def dataset_from_schema(
    data_path: str | Path,
    schema_path: str | Path,
    **kwargs,
) -> SpectrumDataset:
    """Instantiate the correct dataset class by inspecting the schema file.

    Selection rules
    ---------------
    - top-level ``continuous_feature_names``   → RawFeatureDataset
    - otherwise -> PackedDataset

    All extra keyword arguments are forwarded to the chosen constructor.
    """
    schema_type = _detect_schema_type(schema_path)
    cls = {
        "packed": RawFeatureDataset,
        "raw": RawFeatureDataset,
    }[schema_type]
    return cls(data_path, schema_path, **kwargs)
