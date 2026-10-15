"""Feature table for the feature store (Feast offline = parquet, online = Redis).

Because every feature of a target day t only uses data <= t - horizon, the complete
feature vector for each of the next `horizon` days can be computed TODAY. We publish
one row per (series, target day) keyed by `fkey = "<id>|<d>"`; serving then reads
28 keys per request instead of recomputing rolling windows online - the same code
(`build_features`) produced the training matrix, so there is no training/serving skew.
"""

from __future__ import annotations

import pandas as pd

from demandops.config import FeatureConfig
from demandops.data.loader import extend_horizon
from demandops.features.build import build_features, feature_columns

HELPER_COLUMNS = ["price_last_week", "price_item_mean"]
KEY_COLUMNS = ["fkey", "id", "d", "date"]


def fkey(series_id: str, d: int) -> str:
    return f"{series_id}|{d}"


def horizon_feature_table(
    panel: pd.DataFrame, calendar: pd.DataFrame, cfg: FeatureConfig
) -> pd.DataFrame:
    """panel: observed history only. Returns features for d in (last, last + horizon]."""
    last = int(panel.loc[panel["sales"].notna(), "d"].max())
    extended = extend_horizon(panel[panel["d"] <= last], calendar, cfg.horizon)
    feats = build_features(extended, cfg)
    rows = feats[feats["d"] > last].copy()
    rows["fkey"] = rows["id"] + "|" + rows["d"].astype(str)
    rows["event_timestamp"] = panel.loc[panel["d"] == last, "date"].iloc[0]  # forecast origin date
    keep = [*KEY_COLUMNS, "event_timestamp", *feature_columns(cfg), *HELPER_COLUMNS]
    # drop duplicate names (e.g. item_id appears once) while keeping order
    return rows[list(dict.fromkeys(keep))].reset_index(drop=True)
