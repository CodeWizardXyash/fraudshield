"""Scoring logic, independent of the web framework."""

from __future__ import annotations

import threading
from collections import OrderedDict, deque
from pathlib import Path

import pandas as pd

from fraudshield.features.store import FEATURE_NAMES
from fraudshield.models import registry
from fraudshield.monitoring.drift import drift_report
from fraudshield.schema import Transaction

EXPLAINED = [f for f in FEATURE_NAMES if f not in ("category_code", "history_length", "hour")]


class DuplicateConflictError(ValueError):
    """The same transaction_id was reused with a different payload."""


class ScoringService:
    def __init__(self, bundle: dict, metadata: dict, cache_size: int = 10_000,
                 window: int = 5_000) -> None:
        self.model = bundle["model"]
        self.threshold = bundle["threshold"]
        self.profiles = bundle["profiles"]
        self.reference = bundle["reference_values"]
        self.store = bundle["feature_store"]
        self.version = metadata["version"]
        self._seen: OrderedDict[str, tuple[Transaction, dict]] = OrderedDict()
        self._recent: deque = deque(maxlen=window)
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @classmethod
    def from_registry(cls, root: str | Path) -> ScoringService:
        return cls(*registry.load_champion(Path(root)))

    def score(self, txn: Transaction) -> dict:
        """Score and record one transaction. Retries of the same request are idempotent."""
        with self._lock:
            if txn.transaction_id in self._seen:
                original, response = self._seen[txn.transaction_id]
                if original != txn:
                    raise DuplicateConflictError("transaction_id reused with a different payload")
                return {**response, "duplicate": True}
            features = self.store.compute(txn)
            probability, reasons = self._predict(features)
            self.store.update(txn)
            self._recent.append({**features, "score": probability})
            response = {
                "transaction_id": txn.transaction_id,
                "fraud_probability": probability,
                "decision": "review" if probability >= self.threshold else "approve",
                "threshold": self.threshold,
                "reasons": reasons,
                "model_version": self.version,
                "duplicate": False,
            }
            self._seen[txn.transaction_id] = (txn, response)
            if len(self._seen) > self._cache_size:
                self._seen.popitem(last=False)
            return response

    def _predict(self, features: dict) -> tuple[float, list[dict]]:
        """Probability plus one-at-a-time reference substitution reasons.

        Each reason is how much the score falls if that feature is replaced by its
        typical training value. It is a local sensitivity, not a SHAP value.
        """
        changed = [f for f in EXPLAINED if features[f] != self.reference[f]]
        rows = [features] + [{**features, f: self.reference[f]} for f in changed]
        scores = self.model.predict_proba(pd.DataFrame(rows, columns=FEATURE_NAMES))[:, 1]
        probability = float(scores[0])
        if probability < self.threshold:
            return probability, []
        effects = sorted(
            ((probability - float(s), f) for f, s in zip(changed, scores[1:], strict=True)),
            reverse=True,
        )
        return probability, [
            {"feature": f, "value": features[f], "impact": round(e, 6)}
            for e, f in effects[:3] if e > 0
        ]

    def drift(self, min_rows: int = 500) -> dict:
        with self._lock:
            rows = list(self._recent)
        if len(rows) < min_rows:
            return {"status": "insufficient_data", "rows": len(rows), "required": min_rows}
        return drift_report(self.profiles, pd.DataFrame(rows))
