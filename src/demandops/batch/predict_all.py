"""Nightly batch forecast: predict 28 days for ALL series with the champion model."""

from __future__ import annotations

import pandas as pd

from demandops.batch.forecast_table import write_forecasts
from demandops.config import Settings
from demandops.models.lgbm import LGBMForecaster


def predict_all(model: LGBMForecaster, feature_table: pd.DataFrame, version: str) -> pd.DataFrame:
    """feature_table: output of features.store.horizon_feature_table."""
    pred = model.predict_features(feature_table)
    lo, hi = model.interval(pred["pred"].to_numpy())
    out = feature_table[["id", "item_id", "store_id", "d", "date"]].copy()
    out["forecast"] = pred["pred"].to_numpy()
    out["lower"], out["upper"] = lo, hi
    out["model_version"] = version
    out["run_date"] = str(feature_table["event_timestamp"].iloc[0].date())
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.drop(columns=["d"])


def run(
    settings: Settings, model: LGBMForecaster, version: str, feature_table: pd.DataFrame
) -> int:
    return write_forecasts(predict_all(model, feature_table, version), settings.database_url)
