"""DAG 2 - daily_batch_forecast: champion model -> 28-day forecast for every series -> Postgres."""

from __future__ import annotations

from datetime import datetime, timedelta

from pipelines.airflow_compat import ExternalTaskSensor, dag, task

DEFAULT_ARGS = {"retries": 1, "retry_delay": timedelta(minutes=5), "email_on_failure": True}


@dag(
    dag_id="daily_batch_forecast",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["demandops", "batch"],
)
def daily_batch_forecast():
    wait_ingest = ExternalTaskSensor(
        task_id="wait_for_ingest", external_dag_id="daily_ingest", mode="reschedule", timeout=3600
    )

    @task
    def forecast_all() -> dict:
        from demandops.batch import predict_all
        from demandops.config import get_settings
        from demandops.storage import ObjectStore
        from demandops.training.train import load_model

        s = get_settings()
        model, version = load_model(s, alias=s.model_alias)
        table = ObjectStore(s.features_uri, s.s3_endpoint_url).read_parquet(
            "horizon_features.parquet"
        )
        n = predict_all.run(s, model, version, table)
        return {"rows": n, "model_version": version}

    @task
    def reorder_simulation(info: dict) -> dict:
        """Production replay: forecasts stored 28 days ago vs. what happened, forecast-driven reorder
        policy vs. a naive one. Reports "waiting" until the platform has run for 28 days."""
        from demandops.batch.reorder_sim import compare_from_stored_forecasts
        from demandops.config import get_settings
        from demandops.data.live import load_live_panel

        s = get_settings()
        panel = load_live_panel(s, history_days=s.ingest_history_days)
        out = compare_from_stored_forecasts(panel, s.database_url)
        if out is None:
            return {
                "status": "waiting: no matured forecasts yet",
                "model_version": info["model_version"],
            }
        res, n = out
        return {
            k: {"stockout_rate": v.stockout_rate, "overstock_cost": v.overstock_cost}
            for k, v in res.items()
        } | {"model_version": info["model_version"], "n_series": n}

    wait_ingest >> reorder_simulation(forecast_all())


daily_batch_forecast()
