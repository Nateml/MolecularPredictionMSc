from __future__ import annotations

from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel

_PROJECT_ROOT = Path(__file__).parent.parent


class ModelConfig(BaseModel):
    spectrum_length: int = 1000
    hidden_dims: List[int] = [512, 256, 128]
    dropout: float = 0.3
    num_quantiles: int = 3
    regression_hidden_dim: int = 64


class TrainingConfig(BaseModel):
    batch_size: int = 128
    num_epochs: int = 20
    learning_rate: float = 1e-3


class DataConfig(BaseModel):
    dataset_path: Path
    schema_path: Path

    def resolved_dataset_path(self) -> Path:
        return _PROJECT_ROOT / self.dataset_path

    def resolved_schema_path(self) -> Path:
        return _PROJECT_ROOT / self.schema_path


class CheckpointingConfig(BaseModel):
    save_dir: Path = Path("models")

    def resolved_save_dir(self) -> Path:
        return _PROJECT_ROOT / self.save_dir


class ConformalConfig(BaseModel):
    alpha: float = 0.1  # miscoverage level; target coverage = 1 - alpha


class Config(BaseModel):
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig
    checkpointing: CheckpointingConfig = CheckpointingConfig()
    conformal: ConformalConfig = ConformalConfig()


def load_config(path: Path | str | None = None) -> Config:
    config_path = Path(path) if path else _PROJECT_ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config.model_validate(raw)


config: Config = load_config()
