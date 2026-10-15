import numpy as np
import pandas as pd
import pytest

from demandops.features.build import build_features, feature_columns
from demandops.features.lags import add_lags
from demandops.features.price import override_price


def test_lags_shorter_than_horizon_are_rejected(panel):
    with pytest.raises(ValueError, match="leak"):
        add_lags(panel, (7, 28), horizon=28)


def test_lag_values(panel, feat_cfg):
    df = build_features(panel, feat_cfg)
    one = df[df["id"] == df["id"].iloc[0]].set_index("d")
    assert one.loc[100, "lag_28"] == one.loc[72, "sales"]
    assert one.loc[100, "lag_42"] == one.loc[58, "sales"]
    assert np.isnan(one.loc[20, "lag_28"])  # nothing before d_1


def test_rolling_mean_is_shifted_by_horizon(panel, feat_cfg):
    df = build_features(panel, feat_cfg)
    one = df[df["id"] == df["id"].iloc[0]].set_index("d")
    expected = one.loc[100 - 28 - 6 : 100 - 28, "sales"].mean()  # 7 days ending at d-28
    assert one.loc[100, "rmean_7"] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("t", [60, 150, 300])
def test_no_leakage_features_at_t_use_only_data_before_t_minus_horizon(panel, feat_cfg, t):
    """Corrupt every sale from d = t - 27 on: features of day t must not change."""
    base = build_features(panel, feat_cfg)
    corrupted = panel.copy()
    corrupted.loc[corrupted["d"] >= t - feat_cfg.horizon + 1, "sales"] = 9999.0
    after = build_features(corrupted, feat_cfg)

    categorical = ("item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1")
    cols = [c for c in feature_columns(feat_cfg) if c not in categorical]
    a = base[base["d"] == t][cols].reset_index(drop=True)
    b = after[after["d"] == t][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_snap_flag_follows_state(panel, feat_cfg):
    df = build_features(panel, feat_cfg)
    row = df[(df["state_id"] == "TX") & (df["d"] == 50)].iloc[0]
    assert row["snap"] == row["snap_TX"]


def test_price_override_recomputes_dependents(panel, feat_cfg):
    df = build_features(panel, feat_cfg)
    rows = df[(df["id"] == df["id"].iloc[0]) & (df["d"] > 200)].head(5)
    out = override_price(rows, 10.0)
    assert (out["sell_price"] == 10.0).all()
    np.testing.assert_allclose(out["price_vs_item_mean"], 10.0 / rows["price_item_mean"], rtol=1e-5)


def test_price_override_must_be_positive(panel, feat_cfg):
    df = build_features(panel, feat_cfg).head(3)
    with pytest.raises(ValueError):
        override_price(df, 0)


def test_short_history_gives_identical_horizon_features(panel, calendar):
    """The ingest job publishes features from a truncated history. That is only safe if the result
    equals the full-history result, i.e. every feature depends on a bounded window (the reason the
    price mean is a trailing window instead of an expanding mean)."""
    from demandops.config import FeatureConfig
    from demandops.features.store import horizon_feature_table

    cfg = FeatureConfig(price_window=100)
    need = cfg.price_window + cfg.horizon + max(cfg.rolling_windows)
    last = int(panel["d"].max())
    full = panel[panel["d"] <= last - 20]
    short = full[full["d"] > int(full["d"].max()) - need]
    a = horizon_feature_table(full, calendar, cfg)
    b = horizon_feature_table(short, calendar, cfg)
    cols = [c for c in a.columns if c not in ("event_timestamp",)]
    pd.testing.assert_frame_equal(a[cols], b[cols])


def test_price_mean_ignores_prices_inside_the_horizon(panel, feat_cfg):
    t = 300
    base = build_features(panel, feat_cfg)
    changed = panel.copy()
    changed.loc[changed["d"] > t - feat_cfg.horizon, "sell_price"] = 99.0
    after = build_features(changed, feat_cfg)
    a = base[base["d"] == t]["price_item_mean"].reset_index(drop=True)
    b = after[after["d"] == t]["price_item_mean"].reset_index(drop=True)
    pd.testing.assert_series_equal(a, b)
