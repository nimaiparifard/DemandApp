import numpy as np
import pandas as pd
import pytest

from demandops.batch.reorder_sim import compare_policies, simulate
from demandops.data.simulator import Simulator, landing_to_long
from demandops.data.validation import DataQualityError, assert_valid, validate_panel
from demandops.monitoring.drift import DriftReport, build_report, psi, should_retrain
from demandops.storage import ObjectStore


# ---------------- validation ----------------
def test_clean_panel_passes(panel):
    obs = panel[panel["d"] <= 400]
    assert validate_panel(obs).ok


def test_negative_sales_detected(panel):
    bad = panel[panel["d"] <= 400].copy()
    bad.loc[5, "sales"] = -1.0
    rep = validate_panel(bad)
    assert not rep.ok and any("negative" in f or "schema" in f for f in rep.failures)


def test_non_positive_price_detected(panel):
    bad = panel[panel["d"] <= 400].copy()
    bad.loc[7, "sell_price"] = 0.0
    assert not validate_panel(bad).ok


def test_series_missing_latest_day_stops_pipeline(panel):
    obs = panel[panel["d"] <= 400]
    victim = obs["id"].iloc[0]
    bad = obs[~((obs["id"] == victim) & (obs["d"] == 400))]
    with pytest.raises(DataQualityError, match="missing day d_400"):
        assert_valid(bad)


def test_calendar_gap_detected(panel):
    obs = panel[(panel["d"] <= 400) & (panel["d"] != 200)]
    assert any("gaps" in f for f in validate_panel(obs).failures)


# ---------------- drift ----------------
def test_psi_identical_is_zero_and_shift_is_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, a) == pytest.approx(0, abs=1e-6)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.1
    assert psi(a, rng.normal(2, 1, 5000)) > 0.2


def test_psi_handles_constant_reference():
    assert psi(np.zeros(100), np.zeros(100)) == pytest.approx(0, abs=1e-6)
    assert psi(np.zeros(100), np.ones(100)) > 0.2


def test_report_and_retrain_decision():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    same = pd.DataFrame({"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)})
    shifted = pd.DataFrame({"a": rng.normal(3, 1, 2000), "b": rng.normal(3, 1, 2000)})
    assert not should_retrain(build_report(ref, same, ["a", "b"]))
    rep = build_report(ref, shifted, ["a", "b"])
    assert rep.data_drift_score == 1.0 and should_retrain(rep)


def test_error_drift_triggers_retrain_even_without_data_drift():
    rep = DriftReport(feature_psi={"a": 0.01}, forecast_wape=0.9, baseline_wape=0.6)
    assert rep.error_drift_ratio == pytest.approx(1.5) and should_retrain(rep)


# ---------------- reorder simulation ----------------
def test_perfect_forecast_with_safety_stock_has_no_stockouts():
    actual = np.full((4, 28), 5.0)
    res = simulate(actual, actual.copy(), sigma=np.full(4, 1.0))
    assert res.stockout_rate == 0.0


def test_zero_forecast_loses_all_sales():
    actual = np.full((2, 14), 3.0)
    res = simulate(actual, np.zeros_like(actual), sigma=np.zeros(2))
    assert res.stockout_rate == pytest.approx(1.0)


def test_forecast_policy_beats_flat_naive_on_seasonal_demand():
    rng = np.random.default_rng(0)
    weekly = np.array([1, 1, 1, 1, 2, 6, 5], dtype=float)
    actual = np.tile(weekly, 4)[None, :].repeat(20, axis=0) + rng.poisson(0.2, (20, 28))
    forecast = np.tile(weekly, 4)[None, :].repeat(20, axis=0)
    res = compare_policies(actual, forecast, actual.mean(axis=1), sigma=np.full(20, 1.0))
    assert res["forecast"].stockout_rate <= res["naive"].stockout_rate


# ---------------- simulator + storage ----------------
@pytest.fixture
def sim_dir(tmp_path, wide, calendar):
    wide.to_csv(tmp_path / "sales_train_evaluation.csv", index=False)
    cal = calendar.copy()
    cal["date"] = cal["date"].dt.strftime("%Y-%m-%d")
    cal["d"] = "d_" + cal["d"].astype(str)
    cal.to_csv(tmp_path / "calendar.csv", index=False)
    return tmp_path


def test_simulator_releases_one_day_per_tick_and_is_restartable(sim_dir, wide):
    store = ObjectStore(str(sim_dir / "landing"))
    sim = Simulator(store, sim_dir, "sales_train_evaluation.csv", start_day=400)
    assert sim.tick() == "sales_2016-02-05.parquet" or sim.position() == 401
    # a NEW simulator object continues from the stored position
    sim2 = Simulator(store, sim_dir, "sales_train_evaluation.csv", start_day=400)
    assert sim2.position() == 401
    sim2.tick()
    assert sim2.position() == 402
    assert len([f for f in store.list() if f.startswith("sales_")]) == 2


def test_simulator_stops_when_history_is_exhausted(sim_dir):
    store = ObjectStore(str(sim_dir / "landing"))
    sim = Simulator(store, sim_dir, "sales_train_evaluation.csv", start_day=418)
    assert len(sim.catch_up(10)) == 2  # only d_419, d_420 exist
    assert sim.tick() is None


def test_landing_merges_with_base_history(sim_dir, wide):
    store = ObjectStore(str(sim_dir / "landing"))
    sim = Simulator(store, sim_dir, "sales_train_evaluation.csv", start_day=400)
    sim.catch_up(3)
    long = landing_to_long(wide, store, 400)
    assert long["d"].max() == 403 and (long.groupby("id")["d"].count() == 403).all()
    truth = wide.set_index("id")["d_402"]
    got = long[long["d"] == 402].set_index("id")["sales"]
    assert (got == truth.loc[got.index]).all()


def test_simulator_demand_shift_changes_sales(sim_dir):
    store = ObjectStore(str(sim_dir / "landing"))
    sim = Simulator(store, sim_dir, "sales_train_evaluation.csv", start_day=400)
    sim.tick(demand_shift=3.0)
    day = store.read_parquet(
        store.list()[0] if store.list()[0].startswith("sales_") else store.list()[1]
    )
    base = sim.sales["d_401"].sum()
    assert day["sales"].sum() > 1.5 * base


def test_object_store_roundtrip(tmp_path):
    store = ObjectStore(str(tmp_path / "bucket"))
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    store.write_parquet(df, "t.parquet")
    store.write_json({"k": 1}, "s.json")
    pd.testing.assert_frame_equal(store.read_parquet("t.parquet"), df)
    assert store.read_json("s.json") == {"k": 1}
    assert store.exists("t.parquet") and not store.exists("nope")


def test_a_large_level_bias_triggers_retrain_even_when_wape_and_features_look_fine():
    """Regression from a real run: a 4x demand shock gave bias -74 % but WAPE only 0.79 -> 0.88
    (ratio 1.11) and 1 of 6 features drifted, so neither original trigger fired."""
    rep = DriftReport(
        feature_psi={"sales": 0.1, "sell_price": 0.0, "rmean_28": 0.0},
        forecast_wape=0.88,
        forecast_bias=-0.74,
        baseline_wape=0.79,
    )
    assert rep.data_drift_score == 0 and rep.error_drift_ratio < 1.25
    assert should_retrain(rep)


def test_target_drift_triggers_retrain():
    rep = DriftReport(feature_psi={"sales": 0.4, "sell_price": 0.0, "a": 0.0, "b": 0.0})
    assert rep.data_drift_score == 0.25 and should_retrain(rep)


def test_small_bias_and_quiet_features_do_not_trigger():
    rep = DriftReport(
        feature_psi={"sales": 0.02}, forecast_wape=0.8, forecast_bias=-0.04, baseline_wape=0.79
    )
    assert not should_retrain(rep)


def _small_cfgs():
    from demandops.config import FeatureConfig, TrainConfig

    tcfg = TrainConfig(num_boost_round=30, train_days=300)
    tcfg.lgbm_params.update(num_leaves=15, min_data_in_leaf=5)
    return FeatureConfig(), tcfg


def test_holdout_replay_is_out_of_sample_and_returns_valid_rates(panel):
    from demandops.batch.reorder_sim import compare_holdout

    fcfg, tcfg = _small_cfgs()
    res, n = compare_holdout(panel[panel["d"] <= 420], fcfg, tcfg)
    assert n == 6 and set(res) == {"forecast", "naive"}
    for r in res.values():
        assert 0.0 <= r.stockout_rate <= 1.0 and r.overstock_cost >= 0


def test_the_fitted_model_never_sees_the_replayed_window(panel):
    """compare_holdout fits on days <= cutoff and predicts the window from older features, so changing
    the window's actuals must not change the forecasts."""
    from demandops.models.lgbm import LGBMForecaster

    fcfg, tcfg = _small_cfgs()
    obs = panel[panel["d"] <= 420]
    corrupted = obs.copy()
    corrupted.loc[corrupted["d"] > 392, "sales"] = 1000.0
    a, b = LGBMForecaster(fcfg, tcfg), LGBMForecaster(fcfg, tcfg)
    a.fit(obs, 392)
    b.fit(corrupted, 392)
    np.testing.assert_allclose(
        a.predict(obs, 392, 28)["pred"].to_numpy(), b.predict(corrupted, 392, 28)["pred"].to_numpy()
    )


def test_stored_forecast_replay(panel, tmp_path):
    from demandops.batch.forecast_table import write_forecasts
    from demandops.batch.reorder_sim import compare_from_stored_forecasts

    obs = panel[panel["d"] <= 420]
    url = f"sqlite:///{tmp_path / 'f.db'}"
    assert compare_from_stored_forecasts(obs, url) is None  # nothing stored yet

    cutoff = 392
    window = obs[(obs["d"] > cutoff) & (obs["d"] <= 420)]
    stored = pd.DataFrame(
        {
            "id": window["id"].to_numpy(),
            "item_id": window["item_id"].to_numpy(),
            "store_id": window["store_id"].to_numpy(),
            "date": window["date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "forecast": window["sales"].to_numpy(float),  # a perfect forecast
            "lower": 0.0,
            "upper": 0.0,
            "model_version": "1",
            "run_date": obs.loc[obs["d"] == cutoff, "date"].iloc[0].strftime("%Y-%m-%d"),
        }
    )
    write_forecasts(stored, url)
    res, n = compare_from_stored_forecasts(obs, url)
    assert n == 6
    assert res["forecast"].stockout_rate <= res["naive"].stockout_rate
