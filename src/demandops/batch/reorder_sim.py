"""Reorder-policy simulation: forecast-driven policy vs. a naive policy.

Policy (order-up-to, one review per day, lead time L days):
  target stock = sum of demand estimate over the lead time + z * sigma * sqrt(L)
  order        = max(target - inventory position, 0)    (arrives after L days)
Naive policy uses the trailing-28-day mean as the demand estimate.
Metrics: stockout rate (share of demand not served) and overstock units-days
(average end-of-day inventory above a day's demand), converted to cost with `holding_cost`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SimResult:
    stockout_rate: float
    avg_inventory: float
    overstock_cost: float
    lost_sales_units: float


def simulate(
    actual: np.ndarray,
    demand_estimate: np.ndarray,
    sigma: np.ndarray,
    lead_time: int = 3,
    z: float = 1.28,
    holding_cost: float = 0.02,
) -> SimResult:
    """actual / demand_estimate: (n_series x h) daily units; sigma: (n_series,) forecast error std."""
    n, h = actual.shape
    stock = demand_estimate[:, :lead_time].sum(axis=1) + z * sigma * np.sqrt(lead_time)
    pipeline = np.zeros((n, lead_time))  # slot k arrives at the start of day t+1+k; an order placed
    # at the end of day t goes to the last slot, i.e. usable on day t + lead_time
    lost = 0.0
    inv_sum = 0.0
    over = 0.0
    for t in range(h):
        stock = stock + pipeline[:, 0]
        pipeline = np.roll(pipeline, -1, axis=1)
        pipeline[:, -1] = 0.0
        sold = np.minimum(stock, actual[:, t])
        lost += float((actual[:, t] - sold).sum())
        stock = stock - sold
        window = demand_estimate[:, t + 1 : t + 1 + lead_time]
        horizon_demand = (
            window.sum(axis=1) if window.shape[1] else demand_estimate[:, -1] * lead_time
        )
        if window.shape[1] < lead_time:  # beyond forecast horizon: extend with the last estimate
            horizon_demand = horizon_demand + demand_estimate[:, -1] * (lead_time - window.shape[1])
        target = horizon_demand + z * sigma * np.sqrt(lead_time)
        position = stock + pipeline.sum(axis=1)
        order = np.maximum(target - position, 0.0)
        pipeline[:, -1] += order
        inv_sum += float(stock.sum())
        over += float(np.maximum(stock - actual[:, t], 0.0).sum())
    total = float(actual.sum())
    return SimResult(
        stockout_rate=lost / total if total else 0.0,
        avg_inventory=inv_sum / (h * n),
        overstock_cost=over * holding_cost,
        lost_sales_units=lost,
    )


def compare_policies(
    actual: np.ndarray,
    forecast: np.ndarray,
    history_mean: np.ndarray,
    sigma: np.ndarray,
    **kw: Any,
) -> dict[str, SimResult]:
    """forecast-driven vs naive (flat trailing-mean) policy on the same actuals."""
    naive = np.repeat(history_mean[:, None], actual.shape[1], axis=1)
    return {
        "forecast": simulate(actual, forecast, sigma, **kw),
        "naive": simulate(actual, naive, sigma, **kw),
    }


def _replay(
    panel: pd.DataFrame, pred: pd.DataFrame, cutoff: int, last: int, **kw: Any
) -> tuple[dict[str, SimResult], int]:
    """pred: wide frame (id x day) of forecasts for days cutoff+1..last."""
    wide = panel.pivot(index="id", columns="d", values="sales").loc[pred.index]
    actual = wide.loc[:, cutoff + 1 : last].to_numpy(float)
    hist = wide.loc[:, cutoff - 27 : cutoff].to_numpy(float)
    res = compare_policies(actual, pred.to_numpy(float), hist.mean(axis=1), hist.std(axis=1), **kw)
    return res, len(pred)


def compare_holdout(
    panel: pd.DataFrame, feat_cfg: Any, train_cfg: Any, horizon: int = 28, **kw: Any
) -> tuple[dict[str, SimResult], int]:
    """Out-of-sample replay of the last `horizon` observed days.

    A fresh model is fitted on days <= cutoff only, so its forecasts for the window use exactly the
    information available in production at that time. (Replaying with the live champion would be
    in-sample: it was trained through the last day and has seen these days.) sigma is the per-series
    std of demand in the 28 days before the window, a proxy for forecast-error spread.
    """
    from demandops.models.lgbm import LGBMForecaster

    last = int(panel.loc[panel["sales"].notna(), "d"].max())
    cutoff = last - horizon
    model = LGBMForecaster(feat_cfg, train_cfg)
    model.fit(panel[panel["d"] <= last], cutoff)
    pred = model.predict(panel[panel["d"] <= last], cutoff, horizon)
    return _replay(panel, pred.pivot(index="id", columns="d", values="pred"), cutoff, last, **kw)


def compare_from_stored_forecasts(
    panel: pd.DataFrame, database_url: str, horizon: int = 28, **kw: Any
) -> tuple[dict[str, SimResult], int] | None:
    """Production replay: the forecasts the batch DAG stored `horizon` days ago vs what happened.
    None until such forecasts exist (the platform needs to have run for `horizon` days)."""
    from sqlalchemy import create_engine, text

    last = int(panel.loc[panel["sales"].notna(), "d"].max())
    cutoff = last - horizon
    origin = panel.loc[panel["d"] == cutoff, "date"].iloc[0].strftime("%Y-%m-%d")
    try:
        with create_engine(database_url).connect() as conn:
            fc = pd.read_sql(
                text("SELECT id, date, forecast FROM forecasts WHERE run_date = :r"),
                conn,
                params={"r": origin},
            )
    except Exception:
        return None
    if fc.empty:
        return None
    day_of = panel.drop_duplicates("d").assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"))
    fc = fc.merge(day_of[["d", "date"]], on="date")
    pred = fc.pivot(index="id", columns="d", values="forecast").dropna()
    if pred.shape[1] < horizon:
        return None
    return _replay(panel, pred, cutoff, last, **kw)
