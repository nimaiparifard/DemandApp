"""Train + backtest every model, log to MLflow, register the LightGBM model as a challenger."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from demandops.config import Settings
from demandops.data.loader import load_panel
from demandops.data.validation import assert_valid
from demandops.models.base import Forecaster
from demandops.models.lgbm import LGBMForecaster
from demandops.models.naive import SeasonalNaive
from demandops.training import tracking
from demandops.training.backtest import BacktestResult, backtest


@dataclass
class TrainOutcome:
    run_id: str
    model_version: str
    metrics: dict[str, float]  # mean over folds
    last_fold: dict[str, float]  # most recent holdout window (has the key `cutoff`)
    results: dict[str, dict[str, float]]  # model name -> summary


def available_baselines() -> list[Forecaster]:
    models: list[Forecaster] = [SeasonalNaive()]
    try:
        import statsforecast  # noqa: F401

        from demandops.models.ets import AutoETSTopDown

        models.append(AutoETSTopDown())
    except ImportError:
        pass
    return models


def _log_backtest(res: BacktestResult, prefix: str = "cv_") -> None:
    for i, fold in enumerate(res.folds):
        for k in ("wrmsse", "mae", "bias", "wape"):
            mlflow.log_metric(f"fold_{k}", fold[k], step=i)
    for k, v in res.summary().items():
        mlflow.log_metric(f"{prefix}{k}", v)


def train_all(
    settings: Settings, panel: pd.DataFrame | None = None, baselines: bool = True
) -> TrainOutcome:
    """Full training pipeline. Returns the LightGBM run registered under alias `challenger`."""
    fcfg, tcfg = settings.features, settings.train
    if panel is None:
        panel = load_panel(
            settings.demandops_data_dir,
            settings.sales_file,
            settings.demandops_n_series,
            tcfg.seed,
            settings.history_days,
        )
    assert_valid(panel)  # stops here on bad data

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("demandops")
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    common_tags = {
        "git_sha": tracking.git_sha(),
        "dvc_hash": tracking.dvc_hash(),
        "data_fingerprint": tracking.data_fingerprint(panel),
    }
    results: dict[str, dict[str, float]] = {}

    if baselines:
        for model in available_baselines():
            with mlflow.start_run(run_name=model.name, tags=common_tags):
                res = backtest(model, panel, tcfg.n_folds, fcfg.horizon)
                _log_backtest(res)
                results[model.name] = res.summary()

    lgbm = LGBMForecaster(fcfg, tcfg)
    with mlflow.start_run(run_name="lightgbm", tags=common_tags) as run:
        mlflow.log_params(
            {
                **{f"lgbm_{k}": v for k, v in tcfg.lgbm_params.items()},
                "num_boost_round": tcfg.num_boost_round,
                "train_days": tcfg.train_days,
                "n_folds": tcfg.n_folds,
                "horizon": fcfg.horizon,
                "lags": list(fcfg.lags),
                "rolling_windows": list(fcfg.rolling_windows),
                "n_series": panel["id"].nunique(),
                "seed": tcfg.seed,
            }
        )
        res = backtest(lgbm, panel, tcfg.n_folds, fcfg.horizon)
        _log_backtest(res)
        results[lgbm.name] = res.summary()

        # calibrate prediction intervals on the backtest residuals
        preds = pd.concat(res.predictions)
        merged = preds.merge(panel[["id", "d", "sales"]], on=["id", "d"])
        lgbm.calibrate_interval(merged["sales"].to_numpy(), merged["pred"].to_numpy())

        # final model: fit on ALL observed data
        last = int(panel.loc[panel["sales"].notna(), "d"].max())
        lgbm.fit(panel, last)
        with tempfile.TemporaryDirectory() as tmp:
            lgbm.save(Path(tmp) / "model")
            mlflow.log_artifacts(str(Path(tmp) / "model"), artifact_path="model")
        mlflow.log_metric("trained_through_day", last)
        mlflow.log_metric("bias_scale", lgbm.bias_scale)
        version = tracking.register_and_alias(
            client, run.info.run_id, settings.model_name, "challenger"
        )

    return TrainOutcome(run.info.run_id, version, res.summary(), res.folds[-1], results)


def load_model(
    settings: Settings, alias: str | None = None, version: str | None = None
) -> tuple[LGBMForecaster, str]:
    """Download a registered model by explicit version or by alias; returns (model, version)."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    if version is None:
        mv = tracking.get_alias_version(client, settings.model_name, alias or settings.model_alias)
        if mv is None:
            raise LookupError(f"no model version with alias '{alias or settings.model_alias}'")
        version = str(mv.version)
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{settings.model_name}/{version}"
    )
    return LGBMForecaster.load(Path(local)), version


def numeric_summary(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(results).T.assign(bias_abs=lambda d: np.abs(d["bias"]))[
        ["wrmsse", "mae", "bias", "train_seconds"]
    ]
