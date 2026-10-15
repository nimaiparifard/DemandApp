"""Rolling-origin backtest. Time-based splits only - never a random split."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd

from demandops.models.base import Forecaster
from demandops.training.metrics import evaluate_forecast


def fold_cutoffs(last_observed_day: int, n_folds: int, horizon: int) -> list[int]:
    """Oldest first: the last fold predicts exactly the final `horizon` observed days."""
    return [last_observed_day - horizon * k for k in range(n_folds, 0, -1)]


@dataclass
class BacktestResult:
    model: str
    folds: list[dict[str, float]] = field(default_factory=list)
    predictions: list[pd.DataFrame] = field(default_factory=list)
    train_seconds: float = 0.0

    def mean(self, key: str) -> float:
        return float(sum(f[key] for f in self.folds) / len(self.folds))

    def summary(self) -> dict[str, float]:
        return {
            "wrmsse": self.mean("wrmsse"),
            "mae": self.mean("mae"),
            "bias": self.mean("bias"),
            "wape": self.mean("wape"),
            "train_seconds": self.train_seconds,
        }


def backtest(model: Forecaster, panel: pd.DataFrame, n_folds: int, horizon: int) -> BacktestResult:
    """`panel` holds only observed days (no NaN horizon rows)."""
    last = int(panel.loc[panel["sales"].notna(), "d"].max())
    res = BacktestResult(model=model.name)
    for cutoff in fold_cutoffs(last, n_folds, horizon):
        t0 = time.perf_counter()
        model.fit(panel, cutoff)
        pred = model.predict(panel, cutoff, horizon)
        res.train_seconds += time.perf_counter() - t0  # fit + predict (ETS fits lazily)
        scores = evaluate_forecast(panel, pred, cutoff, horizon)
        scores["cutoff"] = float(cutoff)
        res.folds.append(scores)
        res.predictions.append(pred.assign(cutoff=cutoff))
    return res
