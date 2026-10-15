import numpy as np
import pandas as pd
import pytest

from demandops.config import FeatureConfig, TrainConfig
from demandops.models.lgbm import BIAS_SCALE_CLIP, LGBMForecaster
from demandops.training.backtest import backtest


def _model() -> LGBMForecaster:
    tcfg = TrainConfig(num_boost_round=30, train_days=300)
    tcfg.lgbm_params.update(num_leaves=15, min_data_in_leaf=5)
    return LGBMForecaster(FeatureConfig(), tcfg)


def test_scale_is_learned_only_from_matured_forecasts(panel):
    obs = panel[panel["d"] <= 420]
    m = _model()
    m.fit(obs, 336)
    assert m.bias_scale == 1.0  # nothing matured yet
    pred = m.predict(obs, 336, 28)  # window d_337..d_364
    assert m.bias_scale == 1.0  # predicting must not change the scale (no peeking)

    m.fit(obs, 364)  # the window above is now fully observed
    truth = obs.merge(pred[["id", "d"]], on=["id", "d"])["sales"].sum()
    raw = (pred["pred"] / 1.0).sum()
    expected = np.clip(truth / raw, *BIAS_SCALE_CLIP)
    assert m.bias_scale == pytest.approx(expected, rel=1e-6)


def test_forecasts_made_in_the_future_of_cutoff_are_ignored(panel):
    """A prediction whose days are not yet observed at fit time must not move the scale."""
    obs = panel[panel["d"] <= 420]
    m = _model()
    m.fit(obs, 336)
    m.predict(obs, 336, 28)
    m.fit(obs, 340)  # only 3 of the 28 predicted days are <= 340
    after_partial = m.bias_scale
    m.fit(obs, 364)
    assert after_partial != m.bias_scale or after_partial == 1.0


def test_scale_is_clipped(panel):
    obs = panel[panel["d"] <= 420].copy()
    m = _model()
    m.fit(obs, 336)
    m.predict(obs, 336, 28)
    obs.loc[(obs["d"] > 336) & (obs["d"] <= 364), "sales"] *= 50  # absurd level shift
    m.fit(obs, 364)
    assert m.bias_scale == BIAS_SCALE_CLIP[1]


def test_serving_path_applies_the_scale_and_it_survives_save_load(panel, calendar, tmp_path):
    from demandops.features.store import horizon_feature_table

    obs = panel[panel["d"] <= 420]
    m = _model()
    m.fit(obs, 336)
    m.predict(obs, 336, 28)
    m.fit(obs, 364)
    table = horizon_feature_table(obs, calendar, m.feat_cfg)
    scaled = m.predict_features(table)["pred"].to_numpy()
    m.bias_scale = 1.0
    raw = m.predict_features(table)["pred"].to_numpy()
    m.bias_scale = 1.1
    np.testing.assert_allclose(m.predict_features(table)["pred"], raw * 1.1)
    assert scaled.shape == raw.shape

    m.save(tmp_path / "m")
    assert LGBMForecaster.load(tmp_path / "m").bias_scale == pytest.approx(1.1)


def test_backtest_uses_only_past_folds_for_the_correction(panel):
    """Fold 1 is uncorrected, so its predictions equal the raw model output."""
    obs = panel[panel["d"] <= 420]
    m = _model()
    res = backtest(m, obs, n_folds=3, horizon=28)
    first = res.predictions[0]
    raw_first = m._matured[0]
    pd.testing.assert_series_equal(
        first["pred"].reset_index(drop=True),
        raw_first["raw"].reset_index(drop=True),
        check_names=False,
    )
