"""Forecaster interface: every model (naive, ETS, LightGBM, ...) is trained and scored
through the same backtest."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class Forecaster(Protocol):
    name: str

    def fit(self, panel: pd.DataFrame, cutoff: int) -> None:
        """Learn from rows with d <= cutoff. `panel` may contain later rows (future actuals
        and NaN horizon rows); the model must not use them."""
        ...

    def predict(self, panel: pd.DataFrame, cutoff: int, horizon: int) -> pd.DataFrame:
        """Return DataFrame[id, d, pred] for d in (cutoff, cutoff + horizon]."""
        ...
