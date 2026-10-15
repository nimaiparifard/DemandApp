"""Compute the monitoring report from the live panel + the forecast table."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from demandops.config import Settings
from demandops.features.build import build_features
from demandops.monitoring.drift import DriftReport, build_report
from demandops.training import tracking

MONITORED = ["sales", "sell_price", "rmean_28", "rstd_28", "price_change_wk", "price_vs_item_mean"]


def realised_forecast_error(
    panel: pd.DataFrame, database_url: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Join stored forecasts with the actuals that have arrived since. None if nothing to compare."""
    engine = create_engine(database_url)
    try:
        with engine.connect() as conn:
            fc = pd.read_sql(text("SELECT id, date, forecast FROM forecasts"), conn)
    except Exception:
        return None
    if fc.empty:
        return None
    fc["date"] = pd.to_datetime(fc["date"])
    # keep the most recent forecast that was issued for each (id, date)
    joined = fc.merge(panel[["id", "date", "sales"]].dropna(), on=["id", "date"])
    if joined.empty:
        return None
    return joined["sales"].to_numpy(float), joined["forecast"].to_numpy(float)


def compute_drift(
    settings: Settings, panel: pd.DataFrame, baseline_wape: float | None = None
) -> DriftReport:
    observed = panel[panel["sales"].notna()]
    last = int(observed["d"].max())
    feats = build_features(observed, settings.features)
    current = feats[feats["d"] > last - 28]
    reference = feats[(feats["d"] > last - 28 - 90) & (feats["d"] <= last - 56)]
    pair = realised_forecast_error(observed, settings.database_url)
    actual, forecast = pair if pair else (None, None)
    return build_report(reference, current, MONITORED, actual, forecast, baseline_wape)


def champion_info(settings: Settings) -> tuple[str, float | None]:
    """(model version, baseline WAPE) of the champion, from its own backtest. The drift monitor
    compares realised WAPE against this number, so no value has to be configured by hand."""
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    mv = tracking.get_alias_version(client, settings.model_name, "champion")
    if mv is None:
        return "none", None
    metrics = tracking.champion_metrics(client, settings.model_name, "champion") or {}
    return str(mv.version), metrics.get("cv_wape") or None
