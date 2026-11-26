import json
import shutil

import numpy as np
import pytest

from fraudshield.cli import main, read_data
from fraudshield.config import GateConfig
from fraudshield.models import registry
from fraudshield.models.evaluate import best_threshold, total_cost
from fraudshield.monitoring.drift import psi, reference_profile


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(50):
        y = rng.integers(0, 2, 60)
        p = rng.integers(0, 10, 60) / 10  # many ties
        amounts = rng.uniform(1, 100, 60)
        chosen = total_cost(y, amounts, p >= best_threshold(y, p, amounts, 5.0), 5.0)
        brute = min(total_cost(y, amounts, p >= t, 5.0) for t in [*np.unique(p), 1.01])
        assert chosen == pytest.approx(brute)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    profile = reference_profile(rng.normal(size=5000))
    assert psi(profile, rng.normal(size=5000)) < 0.02
    assert psi(profile, rng.normal(2, 1, size=5000)) > 0.25
    with pytest.raises(ValueError):
        psi(profile, [])


def test_training_beats_baselines(trained):
    root, summary = trained
    metrics = summary["test_metrics"]
    assert summary["promoted"]
    assert metrics["pr_auc"] > metrics["rules_baseline_pr_auc"]
    assert metrics["cost_with_model"] < metrics["cost_without_model"]
    assert (root / "reports" / "precision_recall.png").exists()


def test_splits_are_chronological(trained):
    splits = trained[1]["splits"]
    ordered = [splits[n] for n in ("train", "selection", "calibration", "test")]
    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:], strict=False))


def test_corrupted_model_is_refused(trained, tmp_path):
    copy = tmp_path / "models"
    shutil.copytree(trained[0] / "models", copy)
    version = registry.champion_version(copy)
    with (copy / version / registry.BUNDLE).open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(RuntimeError, match="checksum"):
        registry.load_champion(copy)


@pytest.mark.parametrize("score,expected", [(0.1, False), (0.6, False), (0.9, True)])
def test_promotion_gate(tmp_path, score, expected):
    for version, value in (("v1", 0.9), ("v2", score)):
        (tmp_path / version).mkdir()
        (tmp_path / version / registry.METADATA).write_text(
            json.dumps({"selection_pr_auc": value, "data_sha256": "same"})
        )
    assert registry.promote(tmp_path, "v1", GateConfig())[0]
    assert registry.promote(tmp_path, "v2", GateConfig())[0] is expected
    assert registry.champion_version(tmp_path) == ("v2" if expected else "v1")


def test_monitor_cli_round_trip(frame, trained, tmp_path, capsys):
    path = tmp_path / "batch.csv"
    frame.to_csv(path, index=False)
    assert len(read_data(path)) == len(frame)
    shifted = frame.assign(amount=frame.amount * 20)
    shifted.to_csv(path, index=False)
    code = main(["monitor", "--data", str(path), "--models", str(trained[0] / "models")])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "alert"
