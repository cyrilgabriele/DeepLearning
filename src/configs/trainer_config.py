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
    eval_size: float = Field(
        ...,
        description=(
            "Fraction (0, 0.2] of training rows reserved for evaluation; use 0.0 to keep"
            " all rows for training."
        ),
    )

    @field_validator("eval_size")
    @classmethod
    def _check_eval_size(cls, value: float) -> float:
        if not 0 <= value <= 0.2:
            raise ValueError("eval_size must be within [0, 0.2]")
        return value

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
