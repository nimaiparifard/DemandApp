"""Baseline 1: seasonal naive (same weekday of the last observed week)."""

from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalNaive:
    name = "seasonal_naive"

    def fit(self, panel: pd.DataFrame, cutoff: int) -> None:  # nothing to learn
        return None

    def predict(self, panel: pd.DataFrame, cutoff: int, horizon: int) -> pd.DataFrame:
        last_week = panel[(panel["d"] > cutoff - 7) & (panel["d"] <= cutoff)]
        wide = last_week.pivot(index="id", columns="d", values="sales").fillna(0.0)
        base = wide.to_numpy()  # n x 7, columns = d cutoff-6 .. cutoff
        rows = []
        for h in range(1, horizon + 1):
            col = (h - 1) % 7  # d = cutoff + h is the same weekday as d = cutoff - 6 + col
            rows.append(
                pd.DataFrame(
                    {"id": wide.index, "d": cutoff + h, "pred": base[:, col].astype(float)}
                )
            )
        out = pd.concat(rows, ignore_index=True)
        out["pred"] = np.maximum(out["pred"], 0.0)
        return out
