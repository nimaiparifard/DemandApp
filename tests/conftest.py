"""Synthetic M5-shaped data so unit tests run in seconds without the Kaggle files."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from demandops.config import FeatureConfig, Settings, TrainConfig
from demandops.data.loader import attach_context

STORES = {"CA_1": "CA", "TX_1": "TX", "WI_1": "WI"}
ITEMS = [("FOODS_1_001", "FOODS_1", "FOODS"), ("HOBBIES_1_002", "HOBBIES_1", "HOBBIES")]
N_DAYS = 420


def make_calendar(n_days: int = N_DAYS + 40, start: str = "2015-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=n_days)
    cal = pd.DataFrame({"d": np.arange(1, n_days + 1), "date": dates})
    cal["wm_yr_wk"] = 11000 + (cal["d"] - 1) // 7
    cal["wday"] = dates.dayofweek + 1
    cal["month"] = dates.month
    cal["year"] = dates.year
    cal["event_name_1"] = np.where(dates.day == 25, "Xmas-ish", None)
    cal["event_type_1"] = np.where(dates.day == 25, "Religious", None)
    rng = np.random.default_rng(1)
    for st in ("CA", "TX", "WI"):
        cal[f"snap_{st}"] = rng.integers(0, 2, len(cal))
    return cal


def make_wide(n_days: int = N_DAYS, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for item, dept, cat in ITEMS:
        for store, state in STORES.items():
            base = rng.uniform(1.5, 6)
            weekly = np.array([1.0, 0.9, 0.9, 1.0, 1.2, 1.6, 1.5])
            lam = base * np.tile(weekly, n_days // 7 + 1)[:n_days]
            sales = rng.poisson(lam)
            rows.append(
                {
                    "id": f"{item}_{store}_evaluation",
                    "item_id": item, "dept_id": dept, "cat_id": cat, "store_id": store, "state_id": state,
                    **{f"d_{i + 1}": int(sales[i]) for i in range(n_days)},
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


def make_prices(wide: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    out = []
    weeks = cal["wm_yr_wk"].unique()
    for _, r in wide[["item_id", "store_id"]].drop_duplicates().iterrows():
        base = rng.uniform(2, 9)
        for wk in weeks:
            out.append(
                (r.store_id, r.item_id, wk, round(base * (1 + 0.05 * rng.standard_normal()), 2))
            )
    return pd.DataFrame(out, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"])


@pytest.fixture(scope="session")
def calendar() -> pd.DataFrame:
    return make_calendar()


@pytest.fixture(scope="session")
def wide() -> pd.DataFrame:
    return make_wide()


@pytest.fixture(scope="session")
def panel(wide, calendar) -> pd.DataFrame:
    from demandops.data.loader import wide_to_long

    prices = make_prices(wide, calendar)
    return attach_context(wide_to_long(wide), calendar, prices)


@pytest.fixture
def feat_cfg() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture
def fast_settings(tmp_path) -> Settings:
    s = Settings(
        mlflow_tracking_uri=f"sqlite:///{tmp_path / 'mlflow.db'}",
        database_url=f"sqlite:///{tmp_path / 'f.db'}",
        landing_uri=str(tmp_path / "landing"),
        features_uri=str(tmp_path / "features"),
        demandops_data_dir=tmp_path,
    )
    s.train = TrainConfig(
        num_boost_round=30,
        train_days=300,
        lgbm_params={**TrainConfig().lgbm_params, "num_leaves": 15, "min_data_in_leaf": 5},
    )
    return s
