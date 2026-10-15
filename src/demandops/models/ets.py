"""Baseline 2: AutoETS (statsforecast) on an aggregated level, disaggregated top-down.

Fitting 30K item-level ETS models is wasteful; the standard trick is to fit ETS on
store x dept series (70 for the full data) and split each forecast over the items
by their share of the last 28 days of sales.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AutoETSTopDown:
    name = "autoets"

    def __init__(self, season_length: int = 7) -> None:
        self.season_length = season_length
        self._shares: pd.DataFrame | None = None

    def fit(self, panel: pd.DataFrame, cutoff: int) -> None:
        recent = panel[(panel["d"] > cutoff - 28) & (panel["d"] <= cutoff)]
        per_series = (
            recent.groupby(["id", "store_id", "dept_id"], sort=False)["sales"].sum().reset_index()
        )
        group_total = per_series.groupby(["store_id", "dept_id"])["sales"].transform("sum")
        n_in_group = per_series.groupby(["store_id", "dept_id"])["sales"].transform("size")
        per_series["share"] = np.where(
            group_total > 0, per_series["sales"] / group_total, 1 / n_in_group
        )
        self._shares = per_series.set_index("id")[["store_id", "dept_id", "share"]]

    def predict(self, panel: pd.DataFrame, cutoff: int, horizon: int) -> pd.DataFrame:
        from statsforecast import StatsForecast
        from statsforecast.models import AutoETS

        if self._shares is None:
            raise RuntimeError("call fit() first")
        hist = panel[panel["d"] <= cutoff]
        agg = (
            hist.groupby(["store_id", "dept_id", "d", "date"], sort=False)["sales"]
            .sum()
            .reset_index()
        )
        agg["unique_id"] = agg["store_id"] + "|" + agg["dept_id"]
        agg = agg.rename(columns={"date": "ds", "sales": "y"})[["unique_id", "ds", "y"]]

        sf = StatsForecast(models=[AutoETS(season_length=self.season_length)], freq="D", n_jobs=1)
        fc = sf.forecast(df=agg, h=horizon)
        fc["d"] = fc.groupby("unique_id").cumcount() + 1 + cutoff
        fc[["store_id", "dept_id"]] = fc["unique_id"].str.split("|", expand=True)
        fc = fc.rename(columns={"AutoETS": "group_pred"})[
            ["store_id", "dept_id", "d", "group_pred"]
        ]

        shares = self._shares.reset_index()
        out = shares.merge(fc, on=["store_id", "dept_id"])
        out["pred"] = np.maximum(out["group_pred"] * out["share"], 0.0)
        return out[["id", "d", "pred"]]
