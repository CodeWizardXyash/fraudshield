from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from fraudshield.config import Config, DataConfig
from fraudshield.data.simulate import simulate
from fraudshield.models.train import train
from fraudshield.schema import Transaction

SMALL = Config(data=DataConfig(n_customers=300, n_days=60, fraud_episodes_per_day=3, seed=1))


@pytest.fixture(scope="session")
def frame():
    return simulate(SMALL.data)


@pytest.fixture(scope="session")
def trained(frame, tmp_path_factory):
    root = tmp_path_factory.mktemp("artifacts")
    summary = train(frame, SMALL, root / "models", root / "reports")
    return root, summary


def make_txn(index: int = 0, minutes: float = 0, **overrides) -> Transaction:
    values = {
        "transaction_id": f"t{index}",
        "customer_id": "c1",
        "timestamp": datetime(2026, 3, 1, 12, tzinfo=UTC) + timedelta(minutes=minutes),
        "amount": 20.0,
        "merchant_id": "m1",
        "category": "grocery",
        "channel": "pos",
        "country": "IN",
    }
    return Transaction(**{**values, **overrides})
