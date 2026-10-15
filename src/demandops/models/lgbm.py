"""Main model: ONE global LightGBM (tweedie) over all series, direct 28-day horizon.

All features of a target day are computed from data at least `horizon` days
earlier, so a single model predicts any day of the horizon with no recursion.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from demandops.config import FeatureConfig, TrainConfig
from demandops.features.build import build_features, feature_columns, to_model_matrix

BIAS_SCALE_CLIP = (0.85, 1.20)  # a level correction beyond +-15-20 % means something else is wrong


class LGBMForecaster:
    name = "lightgbm"

    def __init__(self, feat_cfg: FeatureConfig, train_cfg: TrainConfig) -> None:
        self.feat_cfg = feat_cfg
        self.train_cfg = train_cfg
        self.booster: lgb.Booster | None = None
        self.categories: dict[str, list[str]] = {}
        self.interval_z: tuple[float, float] = (-1.0, 1.0)  # q05 / q95 of scaled residuals
        # Level correction. Demand grows over time, so any model trained on the past under-forecasts;
        # the factor is sum(actual)/sum(raw prediction) over earlier, already-matured forecasts.
        self.bias_scale: float = 1.0
        self._matured: list[pd.DataFrame] = []  # raw (uncorrected) backtest predictions
        self._features: pd.DataFrame | None = None
        self._features_key: tuple[int, int] | None = None

    # --- helpers --------------------------------------------------------
    def _featurize(self, panel: pd.DataFrame) -> pd.DataFrame:
        key = (id(panel), len(panel))
        if self._features is None or self._features_key != key:
            self._features = build_features(panel, self.feat_cfg)
            self._features_key = key
        return self._features

    # --- Forecaster API -------------------------------------------------
    def _learn_scale(self, panel: pd.DataFrame, cutoff: int) -> None:
        """Update `bias_scale` from forecasts whose actuals are known by `cutoff` - strictly past
        information, so rolling-origin folds stay leak-free (fold 1 runs uncorrected, fold 2 uses
        fold 1's error, ...)."""
        if not self._matured:
            return
        pred = pd.concat(self._matured, ignore_index=True)
        pred = pred[pred["d"] <= cutoff]
        actual = panel.loc[(panel["d"] <= cutoff) & panel["sales"].notna(), ["id", "d", "sales"]]
        joined = pred.merge(actual, on=["id", "d"])
        total = float(joined["raw"].astype("float64").sum())
        if total > 0:
            lo, hi = BIAS_SCALE_CLIP
            ratio = float(joined["sales"].astype("float64").sum()) / total
            self.bias_scale = float(np.clip(ratio, lo, hi))

    def fit(self, panel: pd.DataFrame, cutoff: int) -> None:
        self._learn_scale(panel, cutoff)
        feats = self._featurize(panel)
        lo = cutoff - self.train_cfg.train_days
        mask = (
            (feats["d"] <= cutoff)
            & (feats["d"] > lo)
            & feats["sales"].notna()
            & feats["sell_price"].notna()
        )
        # select only the model columns once (a full-width copy of millions of rows is what
        # dominated peak memory), and let LightGBM free the raw matrix after binning
        cols = [*feature_columns(self.feat_cfg), "sales"]
        train = feats.loc[mask, cols]
        x, self.categories = to_model_matrix(train, self.feat_cfg)
        label = train["sales"].to_numpy()
        del train
        ds = lgb.Dataset(x, label=label, free_raw_data=True)
        params = {**self.train_cfg.lgbm_params, "seed": self.train_cfg.seed, "num_threads": 0}
        self.booster = lgb.train(params, ds, num_boost_round=self.train_cfg.num_boost_round)
        del x, ds

    def predict(self, panel: pd.DataFrame, cutoff: int, horizon: int) -> pd.DataFrame:
        feats = self._featurize(panel)
        rows = feats[(feats["d"] > cutoff) & (feats["d"] <= cutoff + horizon)]
        raw = self._raw(self.prepare(rows))
        self._matured.append(
            pd.DataFrame({"id": rows["id"].to_numpy(), "d": rows["d"].to_numpy(), "raw": raw})
        )
        return pd.DataFrame(
            {"id": rows["id"].to_numpy(), "d": rows["d"].to_numpy(), "pred": raw * self.bias_scale}
        )

    def prepare(self, rows: pd.DataFrame) -> pd.DataFrame:
        """Model matrix for feature rows (fixed categories). Split from `predict_matrix` so a
        server can cache it: features only change when the daily ingest runs."""
        x, _ = to_model_matrix(rows, self.feat_cfg, self.categories)
        return x

    def _raw(self, x: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("model not fitted")
        return np.maximum(self.booster.predict(x), 0.0)

    def predict_matrix(self, x: pd.DataFrame) -> np.ndarray:
        """Corrected point forecasts (what serving and batch return)."""
        return self._raw(x) * self.bias_scale

    def predict_features(self, rows: pd.DataFrame) -> pd.DataFrame:
        """rows: output of build_features restricted to target days."""
        pred = self.predict_matrix(self.prepare(rows))
        return pd.DataFrame({"id": rows["id"].to_numpy(), "d": rows["d"].to_numpy(), "pred": pred})

    # --- prediction intervals ------------------------------------------
    def calibrate_interval(
        self, actual: np.ndarray, pred: np.ndarray, lo: float = 0.05, hi: float = 0.95
    ) -> None:
        """Empirical quantiles of residuals scaled by sqrt(pred + 1) (variance grows with level)."""
        z = (actual - pred) / np.sqrt(pred + 1.0)
        self.interval_z = (float(np.quantile(z, lo)), float(np.quantile(z, hi)))

    def interval(self, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = np.sqrt(pred + 1.0)
        return np.maximum(pred + self.interval_z[0] * s, 0.0), np.maximum(
            pred + self.interval_z[1] * s, 0.0
        )

    # --- persistence (mlflow artifact dir) -----------------------------
    def save(self, path: Path) -> None:
        if self.booster is None:
            raise RuntimeError("model not fitted")
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path / "booster.txt"))
        meta = {
            "categories": self.categories,
            "interval_z": list(self.interval_z),
            "bias_scale": self.bias_scale,
            "feature_config": self.feat_cfg.model_dump(),
            "feature_columns": feature_columns(self.feat_cfg),
        }
        (path / "meta.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, path: Path, train_cfg: TrainConfig | None = None) -> LGBMForecaster:
        meta = json.loads((path / "meta.json").read_text())
        feat_cfg = FeatureConfig(**meta["feature_config"])
        model = cls(feat_cfg, train_cfg or TrainConfig())
        model.booster = lgb.Booster(model_file=str(path / "booster.txt"))
        model.categories = meta["categories"]
        model.interval_z = (meta["interval_z"][0], meta["interval_z"][1])
        model.bias_scale = float(meta.get("bias_scale", 1.0))
        return model
