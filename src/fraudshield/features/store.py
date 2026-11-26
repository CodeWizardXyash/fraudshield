"""Point-in-time feature computation.

The same `FeatureStore` is used to build the training set (by replaying history
in time order) and to serve live requests, which removes the usual source of
training/serving skew. Features for a transaction only use *earlier* ones.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from fraudshield.schema import CATEGORIES, Transaction

FEATURE_NAMES = [
    "amount",
    "amount_zscore",
    "amount_ratio",
    "hour",
    "is_online",
    "category_code",
    "seconds_since_last",
    "count_10m",
    "count_1h",
    "count_24h",
    "amount_sum_24h",
    "merchants_1h",
    "small_count_1h",
    "is_new_merchant",
    "is_new_device",
    "is_new_country",
    "history_length",
]
CATEGORICAL_FEATURES = ["category_code"]
DAY = 86400.0
MAX_GAP = 7 * DAY
SMALL_AMOUNT = 10.0


class OutOfOrderError(ValueError):
    """Raised when a transaction is older than the customer's latest one."""


@dataclass
class CustomerState:
    count: int = 0
    mean_log: float = 0.0
    m2_log: float = 0.0
    mean_amount: float = 0.0
    last_seen: float | None = None
    recent: deque = field(default_factory=deque)  # (epoch, amount, merchant) for 24h
    merchants: set = field(default_factory=set)
    devices: set = field(default_factory=set)
    countries: set = field(default_factory=set)


class FeatureStore:
    """In-memory per-customer state. Not thread-safe; callers must serialise access."""

    def __init__(self) -> None:
        self.customers: dict[str, CustomerState] = {}

    def compute(self, txn: Transaction) -> dict[str, float]:
        """Features for `txn` from past state only. Does not modify the store."""
        state = self.customers.get(txn.customer_id) or CustomerState()
        now = txn.timestamp.timestamp()
        if state.last_seen is not None and now < state.last_seen:
            raise OutOfOrderError(
                f"transaction is older than the latest one for customer {txn.customer_id}"
            )
        day = [item for item in state.recent if now - item[0] <= DAY]
        hour = [item for item in day if now - item[0] <= 3600]
        log_amount = math.log1p(txn.amount)
        known = state.count > 0
        # Floor the deviation so customers with little history do not explode.
        std = max(math.sqrt(state.m2_log / state.count), 0.25) if state.count > 1 else 1.0
        gap = MAX_GAP if state.last_seen is None else min(now - state.last_seen, MAX_GAP)
        return {
            "amount": txn.amount,
            "amount_zscore": (log_amount - state.mean_log) / std if known else 0.0,
            "amount_ratio": txn.amount / state.mean_amount if known else 1.0,
            "hour": float(txn.timestamp.hour),
            "is_online": float(txn.channel == "online"),
            "category_code": float(CATEGORIES.index(txn.category)),
            "seconds_since_last": gap,
            "count_10m": float(sum(now - item[0] <= 600 for item in hour)),
            "count_1h": float(len(hour)),
            "count_24h": float(len(day)),
            "amount_sum_24h": float(sum(item[1] for item in day)),
            "merchants_1h": float(len({item[2] for item in hour})),
            "small_count_1h": float(sum(item[1] < SMALL_AMOUNT for item in hour)),
            "is_new_merchant": float(known and txn.merchant_id not in state.merchants),
            "is_new_device": float(bool(txn.device_id) and known
                                   and txn.device_id not in state.devices),
            "is_new_country": float(known and txn.country not in state.countries),
            "history_length": float(min(state.count, 50)),
        }

    def update(self, txn: Transaction) -> None:
        state = self.customers.setdefault(txn.customer_id, CustomerState())
        now = txn.timestamp.timestamp()
        state.count += 1
        log_amount = math.log1p(txn.amount)
        delta = log_amount - state.mean_log
        state.mean_log += delta / state.count
        state.m2_log += delta * (log_amount - state.mean_log)
        state.mean_amount += (txn.amount - state.mean_amount) / state.count
        state.last_seen = now
        state.recent.append((now, txn.amount, txn.merchant_id))
        while state.recent and now - state.recent[0][0] > DAY:
            state.recent.popleft()
        state.merchants.add(txn.merchant_id)
        state.countries.add(txn.country)
        if txn.device_id:
            state.devices.add(txn.device_id)


def build_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, FeatureStore]:
    """Replay a time-sorted frame, returning its features and the final state."""
    store = FeatureStore()
    rows = []
    columns = list(Transaction.model_fields)
    for record in frame[columns].to_dict("records"):
        # model_construct skips re-validation; validate_frame has checked the batch.
        txn = Transaction.model_construct(**record)
        rows.append(store.compute(txn))
        store.update(txn)
    return pd.DataFrame(rows, columns=FEATURE_NAMES, index=frame.index), store
