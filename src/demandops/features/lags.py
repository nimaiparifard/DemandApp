"""Lag features.

Direct 28-day forecasting: the forecast origin is `horizon` days before the
target day, so any lag shorter than the horizon would use data that does not
exist yet at prediction time. We therefore REQUIRE lag >= horizon. (The classic
"lag 7 / 14" features are replaced by 35 / 42, which keep the same weekday.)
"""

from __future__ import annotations

import pandas as pd


def add_lags(
    df: pd.DataFrame, lags: tuple[int, ...], horizon: int, inplace: bool = False
) -> pd.DataFrame:
    """Add `lag_k` = sales k days before the row's day. df sorted by (id, d), gap-free."""
    bad = [k for k in lags if k < horizon]
    if bad:
        raise ValueError(
            f"lags {bad} are shorter than horizon={horizon}: they would leak the future"
        )
    out = df if inplace else df.copy()
    grouped = out.groupby("id", sort=False)["sales"]
    for k in lags:
        out[f"lag_{k}"] = grouped.shift(k).astype("float32")
    return out
