"""Population Stability Index (PSI) drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd

WARN, ALERT = 0.1, 0.25
# These features change naturally as customer history builds up.
EXCLUDED = {"history_length"}


def reference_profile(values) -> dict:
    """Bin edges and proportions from reference data (quantile bins)."""
    values = np.asarray(values, dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9)))
    return {"edges": edges.tolist(), "proportions": _proportions(values, edges).tolist()}


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def psi(profile: dict, values) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot compute drift on an empty batch")
    expected = np.clip(np.asarray(profile["proportions"]), 1e-4, None)
    actual = np.clip(_proportions(values, np.asarray(profile["edges"])), 1e-4, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def drift_report(profiles: dict[str, dict], batch: pd.DataFrame) -> dict:
    """PSI per profiled column, with an overall ok / warn / alert status."""
    scores = {n: psi(p, batch[n]) for n, p in profiles.items() if n not in EXCLUDED}
    worst = max(scores.values())
    status = "alert" if worst >= ALERT else "warn" if worst >= WARN else "ok"
    ranked = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
    return {"status": status, "max_psi": worst, "rows": len(batch), "psi": ranked}
