"""CLI + live-data + monitoring paths, run for real on a tiny synthetic M5 directory."""

import json

import pytest
from typer.testing import CliRunner

from demandops.batch import predict_all
from demandops.cli import app
from demandops.config import get_settings
from demandops.data.live import load_live_panel
from demandops.data.simulator import Simulator
from demandops.monitoring import exporters
from demandops.monitoring.drift import DriftReport
from demandops.monitoring.run import compute_drift
from demandops.serving.factory import default_service_factory
from demandops.storage import ObjectStore
from demandops.training.train import load_model

from ..conftest import make_calendar, make_prices, make_wide

pytestmark = pytest.mark.integration
runner = CliRunner()


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A complete little deployment on local disk: data dir + sqlite MLflow/DB + folder 'buckets'."""
    data = tmp_path / "data"
    data.mkdir()
    wide, cal = make_wide(), make_calendar()
    wide.to_csv(data / "sales_train_evaluation.csv", index=False)
    out = cal.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    out["d"] = "d_" + out["d"].astype(str)
    out.to_csv(data / "calendar.csv", index=False)
    make_prices(wide, cal).to_csv(data / "sell_prices.csv", index=False)
    for k, v in {
        "DEMANDOPS_DATA_DIR": str(data),
        "DEMANDOPS_N_SERIES": "0",
        "SIMULATOR_START_DAY": "400",
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path / 'mlflow.db'}",
        "DATABASE_URL": f"sqlite:///{tmp_path / 'f.db'}",
        "LANDING_URI": str(tmp_path / "landing"),
        "FEATURES_URI": str(tmp_path / "features"),
        "FEATURE_BACKEND": "parquet",
    }.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def _ok(result):
    assert result.exit_code == 0, result.output + str(result.exception)
    return result.output


def test_full_cli_flow_simulate_retrain_features_batch_drift(env):
    assert "released 3 day(s)" in _ok(runner.invoke(app, ["simulate", "--days", "3"]))
    # fast model: override boosting rounds through settings is not exposed -> use the pipeline default
    out = _ok(runner.invoke(app, ["retrain"]))
    decision = json.loads(out[out.index("{") :])
    assert decision["promoted"] is True and decision["champion_metrics"] is None

    assert "published 168 feature rows (d_404..d_431)" in _ok(
        runner.invoke(app, ["publish-features"])
    )
    assert "wrote 168 rows" in _ok(runner.invoke(app, ["batch-forecast"]))

    report = json.loads(_ok(runner.invoke(app, ["drift"])))
    # PSI on 6 series (6 distinct feature levels) is unstable, so only the structure is asserted here;
    # the real-data behaviour (small PSI without drift) is covered in the unit tests of psi().
    assert set(report["psi"]) >= {"sales", "sell_price"}
    assert isinstance(report["should_retrain"], bool) and report["wape"] == 0.0


def test_evaluate_prints_and_saves_results_table(env, tmp_path):
    out_file = tmp_path / "results.md"
    out = _ok(runner.invoke(app, ["evaluate", "--out", str(out_file)]))
    assert "| seasonal_naive |" in out and "| lightgbm |" in out
    assert out_file.read_text(encoding="utf-8").startswith("Backtest: 6 series")


def test_live_panel_contains_released_days_and_serving_factory_loads_champion(env):
    Simulator(
        ObjectStore(env.landing_uri), env.demandops_data_dir, env.sales_file, start_day=400
    ).catch_up(2)
    panel = load_live_panel(env)
    assert panel["d"].max() == 402 and panel["sales"].notna().all()

    _ok(runner.invoke(app, ["retrain"]))
    _ok(runner.invoke(app, ["publish-features"]))
    svc = default_service_factory(env)()
    assert svc.version == "1"
    fc = svc.forecast("FOODS_1_001", "CA_1", 28)
    assert len(fc) == 28

    model, version = load_model(env, alias="champion")
    table = ObjectStore(env.features_uri).read_parquet("horizon_features.parquet")
    assert predict_all.run(env, model, version, table) == len(table)


def test_drift_detects_a_demand_shift_injected_through_the_simulator(env):
    sim = Simulator(
        ObjectStore(env.landing_uri), env.demandops_data_dir, env.sales_file, start_day=400
    )
    sim.catch_up(19, demand_shift=4.0)  # 4x demand for the last 19 days
    rep = compute_drift(env, load_live_panel(env))
    assert rep.feature_psi["sales"] > 0.2


def test_exporters_build_prometheus_registry_without_a_gateway():
    rep = DriftReport(
        feature_psi={"sales": 0.5, "sell_price": 0.01}, forecast_wape=0.8, baseline_wape=0.5
    )
    text = "".join(
        m.name + str(s.labels) + str(s.value) + "\n"
        for m in exporters.build_registry(rep, "3").collect()
        for s in m.samples
    )
    assert "demandops_data_drift_score" in text and "'feature': 'sales'" in text
    assert "demandops_error_drift_ratio" in text and "1.6" in text


def test_chunked_horizon_table_equals_the_one_shot_table(env):
    """Chunking by series is only an optimisation: the result must be identical."""
    from demandops.data.live import build_horizon_table
    from demandops.data.loader import read_calendar
    from demandops.features.store import horizon_feature_table

    Simulator(
        ObjectStore(env.landing_uri), env.demandops_data_dir, env.sales_file, start_day=400
    ).catch_up(3)
    one_shot = (
        horizon_feature_table(
            load_live_panel(env, history_days=env.ingest_history_days),
            read_calendar(env.demandops_data_dir),
            env.features,
        )
        .sort_values(["id", "d"])
        .reset_index(drop=True)
    )
    chunked = build_horizon_table(env, chunk_series=2)
    assert chunked["id"].nunique() == 6
    import pandas as pd

    pd.testing.assert_frame_equal(one_shot, chunked)
