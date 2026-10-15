"""Calendar features: weekday, month, event type, SNAP flag for the row's state."""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1"]


def add_calendar_features(df: pd.DataFrame, inplace: bool = False) -> pd.DataFrame:
    out = df if inplace else df.copy()
    out["dow"] = out["date"].dt.dayofweek.astype("int8")
    out["month"] = out["date"].dt.month.astype("int8")
    out["dom"] = out["date"].dt.day.astype("int8")
    out["has_event"] = out["event_name_1"].notna().astype("int8")
    out["event_type_1"] = out["event_type_1"].fillna("none")
    snap = np.select(
        [out["state_id"] == "CA", out["state_id"] == "TX", out["state_id"] == "WI"],
        [out["snap_CA"], out["snap_TX"], out["snap_WI"]],
        default=0,
    )
    out["snap"] = snap.astype("int8")
    return out
