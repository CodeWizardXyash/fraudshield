"""Tests for SQLite-backed feature store persistence."""

import os

import pytest

from conftest import make_txn
from fraudshield.features.sqlite_store import SQLiteFeatureStore
from fraudshield.features.store import FEATURE_NAMES


def test_sqlite_store_persists_across_instances(tmp_path):
    db = tmp_path / "state.db"

    # First instance: score and update two transactions
    store1 = SQLiteFeatureStore(db)
    txn1 = make_txn(0, minutes=0)
    txn2 = make_txn(1, minutes=5)
    f1 = store1.compute(txn1)
    store1.update(txn1)
    f2 = store1.compute(txn2)
    store1.update(txn2)
    assert store1.customer_count() == 1
    store1.close()

    # Second instance: should load persisted state
    store2 = SQLiteFeatureStore(db)
    assert store2.customer_count() == 1
    # A third transaction should see the history from the first two
    txn3 = make_txn(2, minutes=10)
    f3 = store2.compute(txn3)
    assert f3["count_1h"] == 2.0  # saw 2 previous transactions within the hour
    assert f3["history_length"] == 2.0
    store2.close()


def test_sqlite_store_multiple_customers(tmp_path):
    db = tmp_path / "multi.db"
    store = SQLiteFeatureStore(db)

    txn_a = make_txn(0, minutes=0, customer_id="alice")
    txn_b = make_txn(1, minutes=1, customer_id="bob")
    store.compute(txn_a)
    store.update(txn_a)
    store.compute(txn_b)
    store.update(txn_b)
    assert store.customer_count() == 2
    store.close()

    # Reload and verify both customers present
    store2 = SQLiteFeatureStore(db)
    assert store2.customer_count() == 2
    store2.close()


def test_sqlite_features_match_memory_store(tmp_path):
    """SQLite store should produce identical features to the in-memory store."""
    from fraudshield.features.store import FeatureStore

    db = tmp_path / "parity.db"
    mem = FeatureStore()
    sql = SQLiteFeatureStore(db)

    txns = [make_txn(i, minutes=i * 5) for i in range(10)]
    for txn in txns:
        f_mem = mem.compute(txn)
        f_sql = sql.compute(txn)
        for key in FEATURE_NAMES:
            assert f_mem[key] == pytest.approx(f_sql[key]), f"mismatch on {key}"
        mem.update(txn)
        sql.update(txn)
    sql.close()


def test_sqlite_store_idempotent_update(tmp_path):
    """Updating the same customer twice should work (upsert, not insert)."""
    db = tmp_path / "upsert.db"
    store = SQLiteFeatureStore(db)
    txn1 = make_txn(0, minutes=0)
    txn2 = make_txn(1, minutes=5)
    store.compute(txn1)
    store.update(txn1)
    store.compute(txn2)
    store.update(txn2)
    assert store.customer_count() == 1  # same customer, upserted
    store.close()
