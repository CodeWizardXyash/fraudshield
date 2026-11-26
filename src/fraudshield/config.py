"""Typed configuration loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(_Strict):
    n_customers: int = Field(2000, ge=10)
    n_days: int = Field(180, ge=20)
    fraud_episodes_per_day: float = Field(4.0, gt=0)
    seed: int = 42


class SplitConfig(_Strict):
    """Chronological split boundaries, as cumulative fractions of the rows."""

    train_end: float = 0.6
    selection_end: float = 0.7
    calibration_end: float = 0.8

    @model_validator(mode="after")
    def _ordered(self) -> SplitConfig:
        if not 0 < self.train_end < self.selection_end < self.calibration_end < 1:
            raise ValueError("splits must satisfy 0 < train < selection < calibration < 1")
        return self


class CostConfig(_Strict):
    review_cost: float = Field(5.0, ge=0, description="Cost of reviewing one alert")


class GateConfig(_Strict):
    """A candidate must pass these checks before it can be promoted."""

    min_pr_auc: float = Field(0.5, ge=0, le=1)
    max_regression: float = Field(0.02, ge=0)


class Config(_Strict):
    data: DataConfig = DataConfig()
    split: SplitConfig = SplitConfig()
    cost: CostConfig = CostConfig()
    gate: GateConfig = GateConfig()
    seed: int = 42


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return Config.model_validate(raw)
