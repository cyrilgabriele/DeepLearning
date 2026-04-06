"""High-level experiment configuration stitched from stage-specific models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, MutableMapping

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .model.model_config import ModelConfig
from .preprocessing.preprocessing_config import PreprocessingConfig
from .train.trainer_config import TrainerConfig
from .tune.tune_config import TuneConfig


class ExperimentConfig(BaseModel):
    """Bundle of trainer, preprocessing, and model settings."""

    trainer: TrainerConfig = Field(..., description="Trainer-level configuration values.")
    preprocessing: PreprocessingConfig = Field(..., description="Preprocessing recipe parameters.")
    model: ModelConfig = Field(..., description="Model preset and architecture knobs.")
    tune: TuneConfig | None = Field(
        default=None,
        description="Optional Optuna sweep configuration for stage 'tune'.",
    )

    model_config = ConfigDict(frozen=True)


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    """Load YAML configuration without allowing runtime overrides."""

    raw_config = _read_yaml(config_path)
    return ExperimentConfig.model_validate(raw_config)


def _read_yaml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text())
    if data is None:
        raise ValueError(f"Config file {config_path} is empty.")
    if not isinstance(data, MutableMapping):
        raise TypeError(f"Config file {config_path} must define a mapping at the top level.")
    return dict(data)
