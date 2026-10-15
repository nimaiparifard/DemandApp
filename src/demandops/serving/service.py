"""Forecast service: model + feature provider -> forecasts with prediction intervals.

Transport-agnostic (FastAPI/KServe wrapper in `app.py`); easy to unit test.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Protocol

import numpy as np
import pandas as pd

from demandops.features.price import override_price
from demandops.models.lgbm import LGBMForecaster


class FeatureProvider(Protocol):
    def get_rows(self, item_id: str, store_id: str, horizon: int) -> pd.DataFrame:
        """Feature rows (store.horizon_feature_table schema) for days 1..horizon, ordered by d."""
        ...


class SeriesNotFound(LookupError):
    pass


class ParquetFeatureProvider:
    """Reads the horizon feature table straight from a DataFrame/parquet (dev, tests, fallback)."""

    def __init__(self, table: pd.DataFrame) -> None:
        self.table = table.sort_values(["id", "d"])
        self._by_id = {k: g for k, g in self.table.groupby("id", sort=False)}

    @classmethod
    def from_parquet(cls, path: str) -> ParquetFeatureProvider:
        return cls(pd.read_parquet(path))

    def get_rows(self, item_id: str, store_id: str, horizon: int) -> pd.DataFrame:
        sid = f"{item_id}_{store_id}_evaluation"
        rows = self._by_id.get(sid)
        if rows is None:
            raise SeriesNotFound(sid)
        return rows.head(horizon)


class FeastFeatureProvider:
    """Online lookup in Feast/Redis: one entity key per (series, target day)."""

    def __init__(self, repo_path: str, first_day: int) -> None:
        from feast import FeatureStore  # lazy: heavy optional dependency

        self.store = FeatureStore(repo_path=repo_path)
        self.first_day = first_day  # d of the first target day (= last observed + 1)

    def get_rows(self, item_id: str, store_id: str, horizon: int) -> pd.DataFrame:
        sid = f"{item_id}_{store_id}_evaluation"
        keys = [{"fkey": f"{sid}|{self.first_day + h}"} for h in range(horizon)]
        feats = [f"horizon_features:{c}" for c in _ONLINE_FEATURES]
        df = self.store.get_online_features(features=feats, entity_rows=keys).to_df()
        if df["lag_28"].isna().all():
            raise SeriesNotFound(sid)
        df["id"] = sid
        df["d"] = [self.first_day + h for h in range(horizon)]
        return df


_ONLINE_FEATURES = [
    "lag_28", "lag_35", "lag_42", "rmean_7", "rstd_7", "rmean_28", "rstd_28", "rmean_56", "rstd_56",
    "sell_price", "price_change_wk", "price_vs_item_mean", "price_last_week", "price_item_mean",
    "dow", "month", "dom", "has_event", "snap",
    "item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1",
]  # fmt: skip


class ForecastService:
    """Model + features -> forecast with intervals.

    The prepared model matrix of a series is cached for `cache_ttl` seconds (features change
    once a day, when the ingest DAG publishes new ones). What-if requests with a price override
    are always computed fresh.
    """

    def __init__(
        self,
        model: LGBMForecaster,
        version: str,
        provider: FeatureProvider,
        cache_ttl: float = 300.0,
        cache_size: int = 20_000,
    ) -> None:
        self.model = model
        self.version = version
        self.provider = provider
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self._cache: OrderedDict[tuple[str, str, int], tuple[float, np.ndarray, pd.DataFrame]] = (
            OrderedDict()
        )
        self._lock = threading.Lock()

    def _matrix(
        self, item_id: str, store_id: str, horizon: int, price_override: float | None
    ) -> tuple[np.ndarray, pd.DataFrame]:
        key = (item_id, store_id, horizon)
        now = time.monotonic()
        if price_override is None and self.cache_ttl > 0:
            with self._lock:
                hit = self._cache.get(key)
                if hit is not None and hit[0] > now:
                    self._cache.move_to_end(key)
                    return hit[1], hit[2]
        rows = self.provider.get_rows(item_id, store_id, horizon).copy()
        if price_override is not None:
            rows = override_price(rows, price_override)
        days, x = rows["d"].to_numpy(), self.model.prepare(rows)
        if price_override is None and self.cache_ttl > 0:
            with self._lock:
                self._cache[key] = (now + self.cache_ttl, days, x)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
        return days, x

    def forecast(
        self, item_id: str, store_id: str, horizon: int, price_override: float | None = None
    ) -> pd.DataFrame:
        """Returns DataFrame[d, pred, lower, upper]."""
        horizon = min(horizon, self.model.feat_cfg.horizon)
        days, x = self._matrix(item_id, store_id, horizon, price_override)
        pred = self.model.predict_matrix(x)
        lo, hi = self.model.interval(pred)
        return pd.DataFrame({"d": days, "pred": pred, "lower": lo, "upper": hi})

    def benchmark_p95_ms(self, item_id: str, store_id: str, n: int = 100) -> float:
        """Latency of one full forecast call (feature fetch + inference), p95 in ms."""
        times = []
        for _ in range(n):
            self._cache.clear()  # measure the cold path: the gate must not be flattered by the cache
            t0 = time.perf_counter()
            self.forecast(item_id, store_id, self.model.feat_cfg.horizon)
            times.append((time.perf_counter() - t0) * 1000)
        return float(np.percentile(times, 95))
