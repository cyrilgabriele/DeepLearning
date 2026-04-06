"""Pydantic model describing trainer-level configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrainerConfig(BaseModel):
    """Paths and experiment bookkeeping for a single run."""

    experiment_name: str = Field(..., description="Label used when logging metrics.")
    train_csv: Path = Field(..., description="Path to Kaggle's train.csv file.")
    test_csv: Optional[Path] = Field(
        None, description="Optional path to Kaggle's test.csv for inference/checkpointing."
    )
    seed: int = Field(42, ge=0, description="Global random seed controlling all stochastic ops.")

    @field_validator("train_csv")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("test_csv")
    @classmethod
    def _expand_optional_path(cls, value: Optional[Path]) -> Optional[Path]:
        if value is None:
            return None
        return value.expanduser().resolve()

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
