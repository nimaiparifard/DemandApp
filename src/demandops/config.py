"""Typed configuration (12-factor: everything overridable by env variables)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

HORIZON = 28


class FeatureConfig(BaseModel):
    """Feature hyper-parameters. Every lag must be >= horizon (see features/lags.py)."""

    horizon: int = HORIZON
    lags: tuple[int, ...] = (28, 35, 42)
    rolling_windows: tuple[int, ...] = (7, 28, 56)
    price_window: int = 365  # days of price history behind `price_vs_item_mean`


class TrainConfig(BaseModel):
    n_folds: int = 3
    train_days: int = 730  # history window used for fitting
    seed: int = 42
    lgbm_params: dict[str, Any] = Field(
        default_factory=lambda: {
            "objective": "tweedie",
            "tweedie_variance_power": 1.05,  # closer to Poisson: less under-forecast bias on sparse series
            "learning_rate": 0.03,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 0.1,
            "verbosity": -1,
        }
    )
    num_boost_round: int = 300


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    demandops_data_dir: Path = Path("data")
    demandops_n_series: int | None = 100  # None / 0 = all 30,490 series
    s3_endpoint_url: str | None = None
    landing_uri: str = "landing"
    features_uri: str = "features"
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    model_name: str = "demand-forecaster"
    model_alias: str = "champion"
    database_url: str = "sqlite:///forecasts.db"
    airflow_url: str = "http://localhost:8080"
    airflow_user: str = "admin"
    airflow_password: str = "admin"
    sales_file: str = "sales_train_evaluation.csv"
    history_days: int = 0  # training history; 0 = all days (bound it on the full 30,490 series)
    ingest_history_days: int = (
        450  # history used to publish features; must be >= price_window + horizon + 56
    )
    simulator_start_day: int = 1913  # days <= this are "history"; later days are replayed live

    features: FeatureConfig = Field(default_factory=FeatureConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)


@lru_cache
def get_settings() -> Settings:
    return Settings()
