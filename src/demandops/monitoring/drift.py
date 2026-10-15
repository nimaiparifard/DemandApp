"""Drift detection.

* data drift     - Population Stability Index (PSI) per numeric feature between a reference
                   window and the current window; the overall score is the share of drifted features.
* forecast drift - the realised error of the latest forecasts (WAPE, bias) vs. the backtest level.

`evidently_report` renders the full HTML report (stored in the S3 store for humans); the numeric
scores that feed Prometheus/alerting come from the dependency-free functions above, so the
alert path never depends on a heavy library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PSI_DRIFT = 0.2  # conventional: <0.1 stable, 0.1-0.2 moderate, >0.2 significant


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # (almost) constant reference: compare "equals the constant" share
        share_ref = float(np.mean(ref == ref[0]))
        share_cur = float(np.mean(cur == ref[0]))
        eps = 1e-4
        pairs = ((share_cur, share_ref), (1 - share_cur, 1 - share_ref))
        return float(sum((a - b) * np.log((a + eps) / (b + eps)) for a, b in pairs))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class DriftReport:
    feature_psi: dict[str, float] = field(default_factory=dict)
    forecast_wape: float = 0.0
    forecast_bias: float = 0.0
    baseline_wape: float | None = None

    @property
    def drifted_features(self) -> list[str]:
        return [k for k, v in self.feature_psi.items() if v >= PSI_DRIFT]

    @property
    def data_drift_score(self) -> float:
        """Share of monitored features whose PSI >= 0.2 (0..1)."""
        return len(self.drifted_features) / len(self.feature_psi) if self.feature_psi else 0.0

    @property
    def error_drift_ratio(self) -> float:
        """Current WAPE relative to baseline WAPE (1.0 = same as in the backtest)."""
        if not self.baseline_wape:
            return 1.0
        return self.forecast_wape / self.baseline_wape


def data_drift(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> dict[str, float]:
    return {c: psi(reference[c].to_numpy(), current[c].to_numpy()) for c in columns}


def forecast_error(actual: np.ndarray, forecast: np.ndarray) -> tuple[float, float]:
    """(WAPE, relative bias)."""
    denom = float(np.sum(np.abs(actual)))
    if denom == 0:
        return 0.0, 0.0
    return float(np.sum(np.abs(actual - forecast)) / denom), float(
        (np.sum(forecast) - np.sum(actual)) / denom
    )


def build_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    actual: np.ndarray | None = None,
    forecast: np.ndarray | None = None,
    baseline_wape: float | None = None,
) -> DriftReport:
    rep = DriftReport(
        feature_psi=data_drift(reference, current, columns), baseline_wape=baseline_wape
    )
    if actual is not None and forecast is not None:
        rep.forecast_wape, rep.forecast_bias = forecast_error(actual, forecast)
    return rep


BIAS_RETRAIN = (
    0.25  # backtests sit within +-5 %, so a persistent +-25 % level error is a regime change
)
TARGET_PSI_RETRAIN = 0.25  # PSI of the sales series itself


def should_retrain(
    rep: DriftReport,
    data_threshold: float = 0.3,
    error_ratio_threshold: float = 1.25,
    bias_threshold: float = BIAS_RETRAIN,
    target_psi_threshold: float = TARGET_PSI_RETRAIN,
) -> bool:
    """Any of: many features drifted, WAPE well above its backtest level, a large level bias, or the
    target itself (sales) drifting. WAPE and the share of drifted features alone are not enough: on
    intermittent retail data WAPE saturates (a 4x demand shock moved it only from 0.79 to 0.88)
    while the bias was -74 %."""
    return (
        rep.data_drift_score >= data_threshold
        or rep.error_drift_ratio >= error_ratio_threshold
        or abs(rep.forecast_bias) >= bias_threshold
        or rep.feature_psi.get("sales", 0.0) >= target_psi_threshold
    )


def evidently_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], path: str
) -> None:
    """Full HTML report for humans. Needs `pip install evidently`."""
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference[columns], current_data=current[columns])
    report.save_html(path)
