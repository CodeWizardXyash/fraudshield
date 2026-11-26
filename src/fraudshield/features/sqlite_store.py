"""SQLite-backed persistence for customer feature state.

Wraps the in-memory FeatureStore so that customer history survives process
restarts.  On startup the store loads all customer states from disk; every
``update()`` call writes through to SQLite so the database is always current.

Usage::

    store = SQLiteFeatureStore("state.db")
    features = store.compute(txn)
    store.update(txn)
    store.close()

Thread safety: SQLite is accessed with ``check_same_thread=False`` and all
writes happen inside the caller's lock (ScoringService._lock), so no extra
locking is needed here.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from fraudshield.features.store import CustomerState, FeatureStore
from fraudshield.schema import Transaction

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS customer_state (
    customer_id   TEXT PRIMARY KEY,
    count         INTEGER NOT NULL,
    mean_log      REAL    NOT NULL,
    m2_log        REAL    NOT NULL,
    mean_amount   REAL    NOT NULL,
    last_seen     REAL,
    recent_json   TEXT    NOT NULL,
    merchants_json TEXT   NOT NULL,
    devices_json  TEXT    NOT NULL,
    countries_json TEXT   NOT NULL
);
"""

_UPSERT = """
INSERT INTO customer_state
    (customer_id, count, mean_log, m2_log, mean_amount, last_seen,
     recent_json, merchants_json, devices_json, countries_json)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(customer_id) DO UPDATE SET
    count=excluded.count, mean_log=excluded.mean_log,
    m2_log=excluded.m2_log, mean_amount=excluded.mean_amount,
    last_seen=excluded.last_seen, recent_json=excluded.recent_json,
    merchants_json=excluded.merchants_json, devices_json=excluded.devices_json,
    countries_json=excluded.countries_json;
"""


def _serialise_state(state: CustomerState) -> dict:
    return {
        "count": state.count,
        "mean_log": state.mean_log,
        "m2_log": state.m2_log,
        "mean_amount": state.mean_amount,
        "last_seen": state.last_seen,
        "recent": list(state.recent),
        "merchants": sorted(state.merchants),
        "devices": sorted(state.devices),
        "countries": sorted(state.countries),
    }


def _deserialise_state(row: tuple) -> CustomerState:
    from collections import deque

    return CustomerState(
        count=row[0],
        mean_log=row[1],
        m2_log=row[2],
        mean_amount=row[3],
        last_seen=row[4],
        recent=deque(json.loads(row[5])),
        merchants=set(json.loads(row[6])),
        devices=set(json.loads(row[7])),
        countries=set(json.loads(row[8])),
    )


class SQLiteFeatureStore(FeatureStore):
    """FeatureStore with write-through SQLite persistence."""

    def __init__(self, db_path: str | Path) -> None:
        super().__init__()
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._load_all()

    def _load_all(self) -> None:
        """Hydrate in-memory state from SQLite on startup."""
        cursor = self._conn.execute(
            "SELECT customer_id, count, mean_log, m2_log, mean_amount, "
            "last_seen, recent_json, merchants_json, devices_json, countries_json "
            "FROM customer_state"
        )
        loaded = 0
        for row in cursor:
            self.customers[row[0]] = _deserialise_state(row[1:])
            loaded += 1
        log.info("loaded %d customer states from %s", loaded, self._db_path)

    def update(self, txn: Transaction) -> None:
        """Update in-memory state and persist to SQLite."""
        super().update(txn)
        state = self.customers[txn.customer_id]
        self._conn.execute(
            _UPSERT,
            (
                txn.customer_id,
                state.count,
                state.mean_log,
                state.m2_log,
                state.mean_amount,
                state.last_seen,
                json.dumps(list(state.recent)),
                json.dumps(sorted(state.merchants)),
                json.dumps(sorted(state.devices)),
                json.dumps(sorted(state.countries)),
            ),
        )
        self._conn.commit()

    def customer_count(self) -> int:
        """Number of customers with persisted state."""
        row = self._conn.execute("SELECT COUNT(*) FROM customer_state").fetchone()
        return row[0] if row else 0

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None  # type: ignore[assignment]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
