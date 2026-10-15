"""Forecast metrics: WRMSSE (official M5 metric), MAE and bias.

WRMSSE = sum over the 12 aggregation levels of  sum_i w_i * RMSSE_i  / 12, where
  RMSSE_i = sqrt( mean_h (y - yhat)^2 / scale_i ),
  scale_i = mean of squared first differences of the series' training history
            (starting at its first non-zero sale),
  w_i     = share of the series in dollar sales of the last 28 training days
            within its level (so weights sum to 1 per level).
When only a sample of series is evaluated, aggregates are those of the sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

# (name, grouping columns); [] = total
LEVELS: list[tuple[str, list[str]]] = [
    ("total", []),
    ("state", ["state_id"]),
    ("store", ["store_id"]),
    ("cat", ["cat_id"]),
    ("dept", ["dept_id"]),
    ("state_cat", ["state_id", "cat_id"]),
    ("state_dept", ["state_id", "dept_id"]),
    ("store_cat", ["store_id", "cat_id"]),
    ("store_dept", ["store_id", "dept_id"]),
    ("item", ["item_id"]),
    ("item_state", ["item_id", "state_id"]),
    ("item_store", ["item_id", "store_id"]),
]


def _aggregation_matrix(meta: pd.DataFrame, cols: list[str]) -> sparse.csr_matrix:
    n = len(meta)
    if not cols:
        return sparse.csr_matrix(np.ones((1, n)))
    codes = meta.groupby(cols, sort=False).ngroup().to_numpy()
    return sparse.csr_matrix((np.ones(n), (codes, np.arange(n))), shape=(codes.max() + 1, n))


def _rmsse_scale(train: np.ndarray) -> np.ndarray:
    """Per-row scale; rows are cut to start at the first non-zero value."""
    scales = np.empty(train.shape[0])
    for i, row in enumerate(train):
        nz = np.flatnonzero(row)
        row = row[nz[0] :] if len(nz) else row
        diff = np.diff(row)
        scales[i] = np.mean(diff**2) if len(diff) else np.nan
    return scales


def wrmsse(
    meta: pd.DataFrame,
    train: np.ndarray,
    actual: np.ndarray,
    forecast: np.ndarray,
    weights_value: np.ndarray,
) -> float:
    """meta: one row per series with the id columns; train (n x T), actual/forecast (n x h);
    weights_value (n): dollar sales of the last 28 training days."""
    total = 0.0
    for _, cols in LEVELS:
        agg = _aggregation_matrix(meta, cols)
        y_train, y, yhat = agg @ train, agg @ actual, agg @ forecast
        w = agg @ weights_value
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1 / len(w))
        scale = _rmsse_scale(np.asarray(y_train))
        mse = np.mean((np.asarray(y) - np.asarray(yhat)) ** 2, axis=1)
        valid = np.isfinite(scale) & (scale > 0)
        rmsse = np.where(valid, np.sqrt(mse / np.where(valid, scale, 1)), 0.0)
        # renormalise weights over valid rows (flat series carry no information)
        wv = np.where(valid, w, 0.0)
        wv = wv / wv.sum() if wv.sum() > 0 else wv
        total += float(np.sum(wv * rmsse))
    return total / len(LEVELS)


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|a-f| / sum a (the drift monitor's baseline)."""
    denom = float(np.sum(np.abs(actual)))
    return float(np.sum(np.abs(actual - forecast)) / denom) if denom else 0.0


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Relative bias: (sum forecast - sum actual) / sum actual. Positive = over-forecast."""
    denom = float(np.sum(actual))
    return float((np.sum(forecast) - denom) / denom) if denom else 0.0


def evaluate_forecast(
    panel: pd.DataFrame, forecast: pd.DataFrame, cutoff: int, horizon: int
) -> dict[str, float]:
    """Score a forecast frame [id, d, pred] against the actuals in `panel` for d in (cutoff, cutoff+h]."""
    ids = sorted(panel["id"].unique())
    meta = panel.drop_duplicates("id").set_index("id").loc[ids].reset_index()

    hist = panel[panel["d"] <= cutoff].pivot(index="id", columns="d", values="sales").loc[ids]
    test_days = list(range(cutoff + 1, cutoff + horizon + 1))
    act = panel[panel["d"].isin(test_days)].pivot(index="id", columns="d", values="sales").loc[ids]
    fc = forecast.pivot(index="id", columns="d", values="pred").loc[ids, test_days]

    last28 = hist.iloc[:, -28:].to_numpy()
    price28 = (
        panel[(panel["d"] > cutoff - 28) & (panel["d"] <= cutoff)]
        .pivot(index="id", columns="d", values="sell_price")
        .loc[ids]
        .fillna(0.0)
        .to_numpy()
    )
    weights = np.sum(last28 * price28, axis=1)

    a, f = act.to_numpy(dtype=float), fc.to_numpy(dtype=float)
    return {
        "wrmsse": wrmsse(meta, hist.to_numpy(dtype=float), a, f, weights),
        "mae": mae(a, f),
        "bias": bias(a, f),
        "wape": wape(a, f),
    }
