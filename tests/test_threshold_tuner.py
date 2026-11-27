"""Tests for the business-cost precision-recall threshold tuner."""

import json

import numpy as np
import pytest

from fraudshield.models.threshold_tuner import ThresholdResult, ThresholdTuner


@pytest.fixture
def sample_data():
    """Create a small deterministic dataset for threshold tests."""
    rng = np.random.default_rng(42)
    n = 200
    y = np.zeros(n, dtype=int)
    y[:20] = 1  # 10% fraud rate
    probability = rng.uniform(0, 0.3, n)
    probability[:20] = rng.uniform(0.4, 1.0, 20)  # frauds get higher scores
    amounts = rng.uniform(10, 500, n)
    return y, probability, amounts


def test_optimal_threshold_reduces_cost(sample_data):
    y, probability, amounts = sample_data
    tuner = ThresholdTuner(review_cost=5.0)
    result = tuner.find_optimal(y, probability, amounts)
    assert isinstance(result, ThresholdResult)
    assert 0 < result.threshold <= 1.01
    assert result.cost_reduction > 0
    assert result.precision > 0
    assert result.recall > 0


def test_sweep_returns_sorted_rows(sample_data):
    y, probability, amounts = sample_data
    tuner = ThresholdTuner(review_cost=5.0)
    rows = tuner.sweep(y, probability, amounts)
    assert len(rows) == 9  # default thresholds
    thresholds = [r["threshold"] for r in rows]
    assert thresholds == sorted(thresholds)
    for row in rows:
        assert 0 <= row["precision"] <= 1
        assert 0 <= row["recall"] <= 1
        assert row["alerts"] >= 0


def test_high_review_cost_raises_threshold(sample_data):
    y, probability, amounts = sample_data
    cheap = ThresholdTuner(review_cost=1.0).find_optimal(y, probability, amounts)
    expensive = ThresholdTuner(review_cost=100.0).find_optimal(y, probability, amounts)
    # With expensive reviews, the optimal threshold should be higher (fewer alerts)
    assert expensive.threshold >= cheap.threshold


def test_missed_fraud_multiplier_lowers_threshold(sample_data):
    y, probability, amounts = sample_data
    base = ThresholdTuner(review_cost=5.0, missed_fraud_multiplier=1.0)
    high = ThresholdTuner(review_cost=5.0, missed_fraud_multiplier=5.0)
    r_base = base.find_optimal(y, probability, amounts)
    r_high = high.find_optimal(y, probability, amounts)
    # With expensive missed fraud, threshold should be lower (catch more)
    assert r_high.threshold <= r_base.threshold


def test_save_report(sample_data, tmp_path):
    y, probability, amounts = sample_data
    tuner = ThresholdTuner(review_cost=5.0)
    path = tmp_path / "threshold_report.json"
    tuner.save_report(y, probability, amounts, path)
    assert path.exists()
    report = json.loads(path.read_text())
    assert "optimal" in report
    assert "sweep" in report
    assert "pr_auc" in report
    assert report["optimal"]["cost_reduction"] > 0


def test_invalid_parameters():
    with pytest.raises(ValueError, match="review_cost"):
        ThresholdTuner(review_cost=-1)
    with pytest.raises(ValueError, match="missed_fraud_multiplier"):
        ThresholdTuner(missed_fraud_multiplier=0)


def test_result_to_dict(sample_data):
    y, probability, amounts = sample_data
    tuner = ThresholdTuner(review_cost=5.0)
    result = tuner.find_optimal(y, probability, amounts)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "threshold" in d
    assert "cost_reduction" in d
