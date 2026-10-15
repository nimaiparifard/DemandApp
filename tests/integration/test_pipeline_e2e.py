"""End-to-end on synthetic data: train -> register -> gates -> promote -> second run keeps champion.

Uses a temp SQLite MLflow, so it needs no external services.
"""

import pytest
from mlflow import MlflowClient

from demandops.training import tracking
from demandops.training.train import load_model, train_all
from pipelines.gates import GateConfig
from pipelines.retrain_flow import run_retrain

pytestmark = pytest.mark.integration


def test_first_model_becomes_champion_and_second_identical_one_does_not(panel, fast_settings):
    obs = panel[panel["d"] <= 420]
    first = run_retrain(fast_settings, panel=obs, p95_latency_ms=20.0)
    assert first.promoted and first.champion_metrics is None

    client = MlflowClient(tracking_uri=fast_settings.mlflow_tracking_uri)
    champ = tracking.get_alias_version(client, fast_settings.model_name, "champion")
    assert champ is not None and str(champ.version) == first.challenger_version

    # identical data + seed -> same accuracy -> fails the "better by >= 1%" gate
    second = run_retrain(fast_settings, panel=obs, p95_latency_ms=20.0)
    assert not second.promoted and second.decision == "kept champion"
    assert any("WRMSSE" in r for r in second.reasons)
    champ_after = tracking.get_alias_version(client, fast_settings.model_name, "champion")
    assert str(champ_after.version) == first.challenger_version  # unchanged


def test_slow_model_is_blocked_by_latency_gate(panel, fast_settings):
    obs = panel[panel["d"] <= 420]
    out = run_retrain(fast_settings, panel=obs, p95_latency_ms=250.0)
    assert not out.promoted and any("latency" in r for r in out.reasons)


def test_provenance_is_logged_and_model_is_loadable(panel, fast_settings):
    obs = panel[panel["d"] <= 420]
    out = train_all(fast_settings, panel=obs, baselines=False)
    run = MlflowClient(tracking_uri=fast_settings.mlflow_tracking_uri).get_run(out.run_id)
    assert {"git_sha", "dvc_hash", "data_fingerprint"} <= set(run.data.tags)
    assert run.data.params["horizon"] == "28" and "cv_wrmsse" in run.data.metrics
    model, version = load_model(fast_settings, version=out.model_version)
    assert version == out.model_version and model.booster is not None


def test_bad_data_stops_training_before_anything_is_registered(panel, fast_settings):
    from demandops.data.validation import DataQualityError

    bad = panel[panel["d"] <= 420].copy()
    bad.loc[3, "sales"] = -5.0
    with pytest.raises(DataQualityError):
        train_all(fast_settings, panel=bad, baselines=False)
    assert (
        tracking.get_alias_version(
            MlflowClient(tracking_uri=fast_settings.mlflow_tracking_uri),
            fast_settings.model_name,
            "challenger",
        )
        is None
    )


def test_strict_gate_config_is_respected(panel, fast_settings):
    obs = panel[panel["d"] <= 420]
    strict = GateConfig(max_abs_bias=1e-9)
    out = run_retrain(fast_settings, panel=obs, p95_latency_ms=1.0, gate_cfg=strict)
    assert not out.promoted and any("bias" in r for r in out.reasons)


def test_champion_baseline_wape_comes_from_its_own_backtest(panel, fast_settings):
    from demandops.monitoring.run import champion_info

    obs = panel[panel["d"] <= 420]
    assert champion_info(fast_settings) == ("none", None)
    run_retrain(fast_settings, panel=obs, p95_latency_ms=10.0)
    version, baseline = champion_info(fast_settings)
    assert version == "1" and baseline is not None and 0 < baseline < 5


def test_gate_uses_the_same_window_score_not_the_stored_training_metrics(
    panel, fast_settings, monkeypatch
):
    """Accuracy is decided by `champion_on_window` (frozen champion scored on the challenger's latest
    holdout window). Controlling that function controls the decision, whatever the metrics stored in
    MLflow when the champion was trained say."""
    import pipelines.retrain_flow as flow

    obs = panel[panel["d"] <= 420]
    run_retrain(fast_settings, panel=obs, p95_latency_ms=10.0)  # becomes champion

    calls = []

    def fake(settings, panel_, cutoff, horizon):
        calls.append((cutoff, horizon))
        return {"wrmsse": 99.0, "bias": 0.0}  # a terrible champion on the shared window

    monkeypatch.setattr(flow, "champion_on_window", fake)
    promoted = run_retrain(fast_settings, panel=obs, p95_latency_ms=10.0)
    assert promoted.promoted and calls == [(392, 28)]  # latest window: last observed day 420 - 28

    monkeypatch.setattr(flow, "champion_on_window", lambda *a: {"wrmsse": 0.0001, "bias": 0.0})
    blocked = run_retrain(fast_settings, panel=obs, p95_latency_ms=10.0)
    assert not blocked.promoted and any("WRMSSE" in r for r in blocked.reasons)


def test_champion_on_window_scores_a_real_frozen_champion(panel, fast_settings):
    from pipelines.retrain_flow import champion_on_window

    obs = panel[panel["d"] <= 420]
    assert champion_on_window(fast_settings, obs, 392, 28) is None  # no champion yet
    run_retrain(fast_settings, panel=obs, p95_latency_ms=10.0)
    scores = champion_on_window(fast_settings, obs, 392, 28)
    assert (
        scores is not None and 0 < scores["wrmsse"] < 5 and {"mae", "bias", "wape"} <= set(scores)
    )


def test_clearly_better_challenger_is_promoted(panel, fast_settings):
    obs = panel[panel["d"] <= 420]
    weak = fast_settings.model_copy(deep=True)
    weak.train.num_boost_round = 1  # a deliberately poor first champion
    first = run_retrain(weak, panel=obs, p95_latency_ms=10.0)
    assert first.promoted
    better = run_retrain(fast_settings, panel=obs, p95_latency_ms=10.0)
    assert better.champion_metrics is not None
    assert better.challenger_metrics["window_wrmsse"] < better.champion_metrics["wrmsse"]
    assert better.promoted, better.reasons
