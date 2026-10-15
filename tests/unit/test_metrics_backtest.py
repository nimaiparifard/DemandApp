import numpy as np
import pandas as pd
import pytest

from demandops.models.naive import SeasonalNaive
from demandops.training.backtest import backtest, fold_cutoffs
from demandops.training.metrics import bias, evaluate_forecast, mae, wrmsse


def _meta(n=1):
    return pd.DataFrame(
        {
            "id": [f"s{i}" for i in range(n)],
            "item_id": [f"i{i}" for i in range(n)],
            "dept_id": ["d"] * n, "cat_id": ["c"] * n, "store_id": ["st"] * n, "state_id": ["CA"] * n,
        }
    )  # fmt: skip


def test_wrmsse_is_zero_for_perfect_forecast():
    train = np.array([[1.0, 2, 3, 2, 1, 2, 3, 4]])
    actual = np.array([[2.0, 3, 2]])
    assert wrmsse(_meta(), train, actual, actual.copy(), np.array([1.0])) == 0.0


def test_wrmsse_single_series_equals_rmsse():
    """With one series all 12 aggregation levels coincide, so WRMSSE == RMSSE (hand computed)."""
    train = np.array([[0.0, 0.0, 1.0, 3.0, 2.0, 4.0]])  # starts at first non-zero: 1,3,2,4
    scale = np.mean(np.diff([1.0, 3.0, 2.0, 4.0]) ** 2)  # (4 + 1 + 4) / 3 = 3
    actual = np.array([[3.0, 5.0]])
    forecast = np.array([[2.0, 3.0]])
    expected = np.sqrt(np.mean((actual - forecast) ** 2) / scale)
    assert wrmsse(_meta(), train, actual, forecast, np.array([1.0])) == pytest.approx(expected)


def test_wrmsse_penalises_worse_forecast():
    meta = _meta(2)
    rng = np.random.default_rng(0)
    train = rng.poisson(3, (2, 60)).astype(float)
    actual = rng.poisson(3, (2, 7)).astype(float)
    good = actual + 0.1
    bad = actual + 3.0
    w = np.array([1.0, 1.0])
    assert wrmsse(meta, train, actual, good, w) < wrmsse(meta, train, actual, bad, w)


def test_mae_and_bias():
    a, f = np.array([2.0, 2.0]), np.array([3.0, 1.0])
    assert mae(a, f) == 1.0
    assert bias(a, f) == 0.0
    assert bias(a, np.array([4.0, 4.0])) == pytest.approx(1.0)


def test_fold_cutoffs_are_time_ordered_and_end_at_last_window():
    assert fold_cutoffs(1941, 3, 28) == [1857, 1885, 1913]


def test_seasonal_naive_repeats_last_week(panel):
    cutoff = 300
    pred = SeasonalNaive().predict(panel, cutoff, 14)
    sid = panel["id"].iloc[0]
    hist = panel[(panel["id"] == sid)].set_index("d")["sales"]
    got = pred[pred["id"] == sid].set_index("d")["pred"]
    assert got.loc[cutoff + 1] == hist.loc[cutoff - 6]
    assert got.loc[cutoff + 8] == hist.loc[cutoff - 6]
    assert got.loc[cutoff + 7] == hist.loc[cutoff]


def test_naive_ignores_data_after_cutoff(panel):
    cutoff = 300
    altered = panel.copy()
    altered.loc[altered["d"] > cutoff, "sales"] = 1e6
    a = SeasonalNaive().predict(panel, cutoff, 28)
    b = SeasonalNaive().predict(altered, cutoff, 28)
    pd.testing.assert_frame_equal(a, b)


def test_backtest_scores_each_fold_against_future_only(panel):
    res = backtest(SeasonalNaive(), panel, n_folds=3, horizon=28)
    assert [int(f["cutoff"]) for f in res.folds] == [336, 364, 392]
    assert all(np.isfinite(f["wrmsse"]) and f["wrmsse"] > 0 for f in res.folds)


def test_evaluate_forecast_perfect(panel):
    cutoff = 392
    truth = panel[(panel["d"] > cutoff) & (panel["d"] <= cutoff + 28)][["id", "d", "sales"]]
    res = evaluate_forecast(panel, truth.rename(columns={"sales": "pred"}), cutoff, 28)
    assert res["wrmsse"] == pytest.approx(0.0, abs=1e-9)
    assert res["mae"] == pytest.approx(0.0, abs=1e-9)
    assert res["wape"] == pytest.approx(0.0, abs=1e-9)
