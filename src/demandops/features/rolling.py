"""Rolling statistics, shifted by `horizon` so a row only sees data up to its forecast origin."""

from __future__ import annotations

import pandas as pd


def add_rolling(
    df: pd.DataFrame, windows: tuple[int, ...], horizon: int, inplace: bool = False
) -> pd.DataFrame:
    """Add `rmean_w` / `rstd_w`: mean/std of sales over the w days ending at day (d - horizon)."""
    out = df if inplace else df.copy()
    shifted = out.groupby("id", sort=False)["sales"].shift(horizon)
    grp = shifted.groupby(out["id"], sort=False)
    for w in windows:
        out[f"rmean_{w}"] = (
            grp.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True).astype("float32")
        )
        out[f"rstd_{w}"] = (
            grp.rolling(w, min_periods=2).std().reset_index(level=0, drop=True).astype("float32")
        )
    return out
