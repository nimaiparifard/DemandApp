"""DAG 4 - monitoring: drift + realised forecast error -> Prometheus Pushgateway.

Prometheus alert rules (infra/k8s-addons/monitoring/alerts.yaml) fire on the pushed gauges;
Alertmanager's webhook then triggers DAG 3 through the Airflow REST API. The HTML report
for humans is written to the S3 store.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from pipelines.airflow_compat import dag, task

DEFAULT_ARGS = {"retries": 1, "retry_delay": timedelta(minutes=5)}


@dag(
    dag_id="monitoring",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["demandops", "monitoring"],
)
def monitoring():
    @task
    def drift_report() -> dict:
        from demandops.config import get_settings
        from demandops.data.live import load_live_panel
        from demandops.monitoring import exporters
        from demandops.monitoring.run import champion_info, compute_drift

        s = get_settings()
        panel = load_live_panel(s, history_days=s.ingest_history_days)
        version, baseline = champion_info(s)
        rep = compute_drift(s, panel, baseline_wape=baseline)
        exporters.push(
            rep, version, os.environ.get("PUSHGATEWAY_URL", "pushgateway.monitoring.svc:9091")
        )
        return {
            "data_drift_score": rep.data_drift_score,
            "drifted": rep.drifted_features,
            "wape": rep.forecast_wape,
            "error_drift_ratio": rep.error_drift_ratio,
        }

    drift_report()


monitoring()
