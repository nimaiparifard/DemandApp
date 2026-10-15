"""Single entry point that builds the model matrix. Used by training, batch AND the
feature-store ingestion, so there is exactly one definition of every feature
(no training/serving skew)."""

from __future__ import annotations

import pandas as pd

from demandops.config import FeatureConfig
from demandops.features.calendar import CATEGORICAL, add_calendar_features
from demandops.features.lags import add_lags
from demandops.features.price import PRICE_FEATURES, add_price_features
from demandops.features.rolling import add_rolling

NUMERIC_CALENDAR = ["dow", "month", "dom", "has_event", "snap"]


def feature_columns(cfg: FeatureConfig) -> list[str]:
    lag_cols = [f"lag_{k}" for k in cfg.lags]
    roll_cols = [f"{s}_{w}" for w in cfg.rolling_windows for s in ("rmean", "rstd")]
    return [*lag_cols, *roll_cols, *PRICE_FEATURES, *NUMERIC_CALENDAR, *CATEGORICAL]


def build_features(panel: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """panel: sorted by (id, d), gap-free (history + optional NaN-sales horizon rows)."""
    # one copy, then every step mutates it: at 30K series the old copy-per-step cost 4x the memory
    df = panel.sort_values(["id", "d"]).reset_index(drop=True)
    add_lags(df, cfg.lags, cfg.horizon, inplace=True)
    add_rolling(df, cfg.rolling_windows, cfg.horizon, inplace=True)
    add_price_features(df, cfg.horizon, cfg.price_window, inplace=True)
    add_calendar_features(df, inplace=True)
    return df


def to_model_matrix(
    df: pd.DataFrame, cfg: FeatureConfig, categories: dict[str, list[str]] | None = None
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Select feature columns; categoricals become pandas `category` with FIXED categories
    (so train and predict encode identically)."""
    x = df[feature_columns(cfg)].copy()
    cats = categories or {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c], categories=cats[c])
    return x, cats
