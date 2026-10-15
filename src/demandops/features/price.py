"""Price features. Prices are set in advance by the retailer, so they are known for the target day."""

from __future__ import annotations

import numpy as np
import pandas as pd

PRICE_FEATURES = ["sell_price", "price_change_wk", "price_vs_item_mean"]


def add_price_features(
    df: pd.DataFrame, horizon: int, window: int = 365, inplace: bool = False
) -> pd.DataFrame:
    """sell_price, change vs. last week's price, and price relative to the item's mean price.

    The mean is a trailing `window`-day mean shifted by `horizon` (only history known at the
    forecast origin). A fixed window - unlike an expanding mean - gives the same value
    whether the panel holds 450 or 1,900 days of history, so the ingest job can use a short
    history without creating training/serving skew. `price_item_mean` / `price_last_week`
    are kept as helper columns so the serving layer can recompute the 3 features for a
    what-if `price_override`.
    """
    out = df if inplace else df.copy()
    g = out.groupby("id", sort=False)["sell_price"]
    out["price_last_week"] = g.shift(7).astype("float32")
    trailing = g.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
    out["price_item_mean"] = (
        trailing.groupby(out["id"], sort=False).shift(horizon).astype("float32")
    )
    return apply_price_features(out, inplace=True)


def apply_price_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = df if inplace else df.copy()
    out["price_change_wk"] = (
        (out["sell_price"] - out["price_last_week"]) / out["price_last_week"]
    ).astype("float32")
    out["price_vs_item_mean"] = (out["sell_price"] / out["price_item_mean"]).astype("float32")
    return out


def override_price(features: pd.DataFrame, new_price: float) -> pd.DataFrame:
    """What-if: replace sell_price and recompute the dependent features."""
    if not np.isfinite(new_price) or new_price <= 0:
        raise ValueError("price_override must be > 0")
    out = features.copy()
    out["sell_price"] = np.float32(new_price)
    return apply_price_features(out)
