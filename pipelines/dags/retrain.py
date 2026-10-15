"""DAG 3 - retrain: weekly, or triggered by the drift alert (Alertmanager -> Airflow REST API).

train challenger -> backtest -> gates (accuracy, bias, latency, data quality) ->
register -> promote -> open GitOps PR (canary rollout happens when the PR is merged).
Otherwise the DAG ends with "kept champion" and an email.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

from pipelines.airflow_compat import dag, get_current_context, task

DEFAULT_ARGS = {"retries": 0, "email_on_failure": True}


@dag(
    dag_id="retrain",
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    dagrun_timeout=timedelta(hours=4),
    tags=["demandops", "mlops"],
)
def retrain():
    @task
    def train_and_gate() -> dict:
        from demandops.config import get_settings
        from pipelines.retrain_flow import as_dict, run_retrain

        decision = run_retrain(get_settings())
        return as_dict(decision)

    @task.short_circuit
    def promoted(decision: dict) -> bool:
        return bool(decision["promoted"])

    @task
    def open_gitops_pr(decision: dict) -> str:
        from pipelines.gitops import open_model_bump_pr

        m = decision["challenger_metrics"]
        body = (
            f"Challenger **v{decision['challenger_version']}** passed all gates.\n\n"
            f"| metric | challenger | champion |\n|---|---|---|\n"
            f"| WRMSSE (same holdout window) | {m['window_wrmsse']:.4f} | {(decision['champion_metrics'] or {}).get('wrmsse', float('nan')):.4f} |\n"
            f"| bias | {m['bias']:+.2%} (mean of folds) | {(decision['champion_metrics'] or {}).get('bias', float('nan')):+.2%} |\n"
            f"| p95 latency | {decision['p95_latency_ms']:.1f} ms | - |\n"
        )
        return open_model_bump_pr(
            repo=os.environ["GITOPS_REPO"],
            token=os.environ["GITOPS_TOKEN"],
            new_version=decision["challenger_version"],
            api_url=os.environ.get("GITOPS_API_URL", "https://api.github.com"),
            body=body,
        )

    @task(trigger_rule="all_done")
    def notify(decision: dict) -> None:
        from airflow.utils.email import send_email

        ctx = get_current_context()
        trigger = (ctx["dag_run"].conf or {}).get("reason", "schedule")
        send_email(
            to=os.environ.get("ALERT_EMAIL_TO", "root@localhost"),
            subject=f"[DemandOps] retrain: {decision['decision']} (v{decision['challenger_version']})",
            html_content=f"<p>trigger: {trigger}</p><pre>{decision}</pre>",
        )

    d = train_and_gate()
    gate = promoted(d)
    pr = open_gitops_pr(d)
    gate >> pr
    d >> notify(d)


retrain()
