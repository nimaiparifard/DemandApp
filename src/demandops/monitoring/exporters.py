"""Push drift metrics to a Prometheus Pushgateway (the monitoring DAG is short-lived, so it cannot
be scraped). Alert rules in infra/helm/demandops/templates/prometheusrule.yaml fire on these gauges."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from demandops.monitoring.drift import DriftReport


def build_registry(rep: DriftReport, model_version: str) -> CollectorRegistry:
    reg = CollectorRegistry()

    def gauge(name: str, doc: str, value: float) -> None:
        Gauge(name, doc, ["model_version"], registry=reg).labels(model_version).set(value)

    gauge("demandops_data_drift_score", "Share of features with PSI >= 0.2", rep.data_drift_score)
    gauge("demandops_forecast_wape", "WAPE of the latest realised forecasts", rep.forecast_wape)
    gauge("demandops_forecast_bias", "Relative bias of the latest forecasts", rep.forecast_bias)
    gauge("demandops_error_drift_ratio", "Current WAPE / backtest WAPE", rep.error_drift_ratio)

    psi_gauge = Gauge(
        "demandops_feature_psi", "PSI per feature", ["feature", "model_version"], registry=reg
    )
    for name, value in rep.feature_psi.items():
        psi_gauge.labels(name, model_version).set(value)
    return reg


def push(
    rep: DriftReport, model_version: str, gateway: str, job: str = "demandops_monitoring"
) -> None:
    push_to_gateway(gateway, job=job, registry=build_registry(rep, model_version))
