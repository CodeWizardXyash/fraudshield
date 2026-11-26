"""The transaction contract shared by the simulator, training and the API."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CATEGORIES = (
    "grocery",
    "restaurant",
    "fuel",
    "retail",
    "electronics",
    "travel",
    "digital_goods",
    "gift_cards",
)


class Transaction(BaseModel):
    """One card transaction, as known at authorisation time (no label)."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    transaction_id: str = Field(min_length=1, max_length=64)
    customer_id: str = Field(min_length=1, max_length=64)
    timestamp: datetime
    amount: float = Field(gt=0, le=1_000_000, allow_inf_nan=False)
    merchant_id: str = Field(min_length=1, max_length=64)
    category: Literal[CATEGORIES]  # type: ignore[valid-type]
    channel: Literal["pos", "online"]
    country: str = Field(pattern=r"^[A-Z]{2}$")
    device_id: str = Field("", max_length=64)

    @field_validator("timestamp")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _online_needs_device(self) -> Transaction:
        if self.channel == "online" and not self.device_id:
            raise ValueError("online transactions require device_id")
        return self


TRANSACTION_COLUMNS = list(Transaction.model_fields)


def validate_frame(frame: pd.DataFrame) -> None:
    """Fast batch checks for training data. Raises ValueError on the first failure."""
    required = [*TRANSACTION_COLUMNS, "is_fraud"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if frame.empty:
        raise ValueError("dataset is empty")
    if frame[required].isna().any().any():
        raise ValueError("dataset contains null values")
    checks = {
        "transaction_id must be unique": frame["transaction_id"].is_unique,
        "amount must be positive": (frame["amount"] > 0).all(),
        "unknown category": frame["category"].isin(CATEGORIES).all(),
        "unknown channel": frame["channel"].isin(["pos", "online"]).all(),
        "is_fraud must be 0 or 1": frame["is_fraud"].isin([0, 1]).all(),
        "rows must be sorted by timestamp": frame["timestamp"].is_monotonic_increasing,
    }
    for message, passed in checks.items():
        if not passed:
            raise ValueError(message)
