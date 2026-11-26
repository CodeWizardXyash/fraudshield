import pandas as pd
import pytest

from conftest import make_txn
from fraudshield.features.store import FEATURE_NAMES, FeatureStore, OutOfOrderError, build_features
from fraudshield.schema import Transaction


def test_first_transaction_has_neutral_history():
    features = FeatureStore().compute(make_txn())
    assert list(features) == FEATURE_NAMES
    assert features["count_24h"] == 0
    assert features["is_new_country"] == 0
    assert features["amount_ratio"] == 1


def test_compute_does_not_mutate_state():
    store = FeatureStore()
    store.compute(make_txn())
    assert not store.customers


def test_velocity_windows():
    store = FeatureStore()
    for i, minute in enumerate([0, 25 * 60, 26 * 60 - 5]):
        store.update(make_txn(i, minute))
    features = store.compute(make_txn(9, 26 * 60))
    assert (features["count_10m"], features["count_1h"], features["count_24h"]) == (1, 2, 2)
    assert features["seconds_since_last"] == 300


def test_novelty_flags():
    store = FeatureStore()
    store.update(make_txn())
    features = store.compute(
        make_txn(1, 5, merchant_id="m2", country="US", channel="online", device_id="d9")
    )
    assert features["is_new_merchant"] == features["is_new_country"] == 1
    assert features["is_new_device"] == 1


def test_out_of_order_is_rejected():
    store = FeatureStore()
    store.update(make_txn(0, 10))
    with pytest.raises(OutOfOrderError):
        store.compute(make_txn(1, 0))


def test_no_future_leakage(frame):
    """Changing or removing later rows must not change earlier features."""
    head = frame.iloc[:2000]
    full, _ = build_features(frame.iloc[:4000])
    prefix, _ = build_features(head)
    pd.testing.assert_frame_equal(full.iloc[:2000], prefix)


def test_label_does_not_affect_features(frame):
    head = frame.iloc[:2000]
    flipped = head.assign(is_fraud=1 - head.is_fraud, scenario="x")
    pd.testing.assert_frame_equal(build_features(head)[0], build_features(flipped)[0])


def test_offline_online_parity(frame):
    """Batch replay and validated one-at-a-time serving give identical features."""
    head = frame.iloc[:1500]
    batch, _ = build_features(head)
    store, rows = FeatureStore(), []
    for record in head[list(Transaction.model_fields)].to_dict("records"):
        record["timestamp"] = record["timestamp"].isoformat()
        txn = Transaction.model_validate(record)
        rows.append(store.compute(txn))
        store.update(txn)
    pd.testing.assert_frame_equal(batch, pd.DataFrame(rows, columns=FEATURE_NAMES))
