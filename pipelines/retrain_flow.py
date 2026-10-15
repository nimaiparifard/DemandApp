"""Train -> backtest -> gates -> promote. Plain functions so Airflow tasks, the CLI and tests share them."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd
from mlflow import MlflowClient

from demandops.config import Settings
from demandops.data.loader import read_calendar
from demandops.data.validation import validate_panel
from demandops.features.build import build_features
from demandops.features.store import horizon_feature_table
from demandops.serving.service import ForecastService, ParquetFeatureProvider
from demandops.training import tracking
from demandops.training.metrics import evaluate_forecast
from demandops.training.train import TrainOutcome, load_model, train_all
from pipelines.gates import GateConfig, GateResult, evaluate_gates

log = logging.getLogger(__name__)


@dataclass
class RetrainDecision:
    challenger_version: str
    promoted: bool
    decision: str
    reasons: list[str]
    challenger_metrics: dict[str, float]
    champion_metrics: dict[str, float] | None
    p95_latency_ms: float


def measure_staging_latency(settings: Settings, panel: pd.DataFrame, version: str) -> float:
    """In-process p95 of one full forecast call for the challenger. In the cluster the DAG may
    instead pass the p95 measured against the staging InferenceService (see retrain.py)."""
    model, _ = load_model(settings, version=version)
    table = horizon_feature_table(
        panel, read_calendar(settings.demandops_data_dir), settings.features
    )
    svc = ForecastService(model, version, ParquetFeatureProvider(table))
    first = table.iloc[0]
    return svc.benchmark_p95_ms(first["item_id"], first["store_id"], n=50)


def champion_on_window(
    settings: Settings, panel: pd.DataFrame, cutoff: int, horizon: int
) -> dict[str, float] | None:
    """Score the frozen champion on the SAME holdout window as the challenger.

    Comparing against the metrics stored when the champion was trained would compare different
    time windows (a quiet month vs. a demand shock), which is meaningless. The champion's features
    use its own feature config; features are leak-safe, so it can predict any past window. If the
    champion was trained on part of the window the comparison favours the champion (conservative).
    Returns None when there is no champion yet.
    """
    try:
        model, _ = load_model(settings, alias="champion")
    except LookupError:
        return None
    feats = build_features(panel, model.feat_cfg)
    rows = feats[(feats["d"] > cutoff) & (feats["d"] <= cutoff + horizon)]
    return evaluate_forecast(panel, model.predict_features(rows), cutoff, horizon)


def decide(
    settings: Settings,
    panel: pd.DataFrame,
    outcome: TrainOutcome,
    dq_failures: int,
    p95_latency_ms: float,
    gate_cfg: GateConfig = GateConfig(),
) -> tuple[GateResult, dict[str, float] | None]:
    """Accuracy is compared on the latest holdout window (challenger vs frozen champion);
    bias is the challenger's mean over all folds."""
    horizon = settings.features.horizon
    champ = champion_on_window(settings, panel, int(outcome.last_fold["cutoff"]), horizon)
    challenger = {"wrmsse": outcome.last_fold["wrmsse"], "bias": outcome.metrics["bias"]}
    result = evaluate_gates(challenger, champ, p95_latency_ms, dq_failures, gate_cfg)
    return result, champ


def run_retrain(
    settings: Settings,
    panel: pd.DataFrame | None = None,
    p95_latency_ms: float | None = None,
    gate_cfg: GateConfig = GateConfig(),
) -> RetrainDecision:
    from demandops.data.live import load_live_panel

    panel = panel if panel is not None else load_live_panel(settings)
    dq = validate_panel(panel)
    outcome = train_all(settings, panel=panel)
    latency = (
        p95_latency_ms
        if p95_latency_ms is not None
        else measure_staging_latency(settings, panel, outcome.model_version)
    )
    gate, champ = decide(settings, panel, outcome, len(dq.failures), latency, gate_cfg)

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    if gate.passed:
        tracking.promote(client, settings.model_name, "challenger", "champion")
        log.info("promoted version %s to champion", outcome.model_version)
    else:
        log.warning("kept champion: %s", "; ".join(gate.reasons))
    return RetrainDecision(
        challenger_version=outcome.model_version,
        promoted=gate.passed,
        decision=gate.decision,
        reasons=gate.reasons,
        challenger_metrics={**outcome.metrics, "window_wrmsse": outcome.last_fold["wrmsse"]},
        champion_metrics=champ,
        p95_latency_ms=latency,
    )


def as_dict(d: RetrainDecision) -> dict:
    return asdict(d)
