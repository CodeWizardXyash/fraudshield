import pandas as pd
import pytest
from pydantic import ValidationError

from conftest import SMALL, make_txn
from fraudshield.config import SplitConfig, load_config
from fraudshield.data.simulate import simulate
from fraudshield.schema import validate_frame


def test_simulator_is_deterministic(frame):
    pd.testing.assert_frame_equal(frame, simulate(SMALL.data))


def test_simulated_data_is_valid_and_imbalanced(frame):
    validate_frame(frame)
    assert 0 < frame.is_fraud.mean() < 0.1


@pytest.mark.parametrize(
    "column,value",
    [("amount", -1.0), ("category", "unknown"), ("channel", "fax"), ("is_fraud", 2)],
)
def test_bad_batches_are_rejected(frame, column, value):
    broken = frame.copy()
    broken.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_frame(broken)


def test_unsorted_batch_is_rejected(frame):
    with pytest.raises(ValueError, match="sorted"):
        validate_frame(frame.iloc[::-1])


@pytest.mark.parametrize(
    "overrides",
    [{"amount": 0}, {"amount": float("nan")}, {"country": "india"}, {"channel": "online"},
     {"extra_field": 1}],
)
def test_bad_transactions_are_rejected(overrides):
    with pytest.raises(ValidationError):
        make_txn(**overrides)


def test_config_loads_and_rejects_bad_splits():
    assert load_config("configs/default.yaml").data.n_days == 180
    with pytest.raises(ValidationError):
        SplitConfig(train_end=0.8, selection_end=0.7)
