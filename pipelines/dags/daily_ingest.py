"""DAG 1 - daily_ingest: validate -> build features -> publish to the feature store.

Runs on KubernetesExecutor (one pod per task). Imports live inside tasks so DAG parsing stays fast.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from pipelines.airflow_compat import dag, task

DEFAULT_ARGS = {"retries": 2, "retry_delay": timedelta(minutes=2), "email_on_failure": True}


@dag(
    dag_id="daily_ingest",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=DEFAULT_ARGS,
    tags=["demandops", "ingest"],
)
def daily_ingest():
    @task
    def validate() -> int:
        """Great-Expectations-style gate. DataQualityError fails the task -> DAG stops + alert."""
        from demandops.config import get_settings
        from demandops.data.live import load_live_panel
        from demandops.data.validation import assert_valid

        s = get_settings()
        panel = load_live_panel(s, history_days=s.ingest_history_days)
        assert_valid(panel)
        return int(panel.loc[panel["sales"].notna(), "d"].max())

    @task
    def build_and_publish(last_day: int) -> str:
        """Build the 28-day horizon feature table (same code as training), chunked by series so
        memory stays bounded on all 30,490 series, and write it to the S3 store."""
        from demandops.config import get_settings
        from demandops.data.live import build_horizon_table
        from demandops.storage import ObjectStore

        s = get_settings()
        table = build_horizon_table(s)
        assert int(table["d"].min()) == last_day + 1
        ObjectStore(s.features_uri, s.s3_endpoint_url).write_parquet(
            table, "horizon_features.parquet"
        )
        return f"{len(table)} rows for d_{last_day + 1}..d_{int(table['d'].max())}"

    @task
    def materialize(_msg: str) -> None:
        """offline (S3 parquet) -> online (Redis)."""
        from datetime import UTC

        from feast import FeatureStore

        # definitions are registered by the `feast apply` init job (infra/k8s-addons/feast)
        store = FeatureStore(repo_path="/opt/airflow/feature_repo")
        store.materialize_incremental(end_date=datetime.now(UTC))

    materialize(build_and_publish(validate()))


daily_ingest()
