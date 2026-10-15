"""Alertmanager webhook -> Airflow REST API (triggers the `retrain` DAG).

Alertmanager can only POST its own JSON payload, but Airflow's API needs a specific body and
authentication, so this ~60-line bridge sits in between. Only alerts labelled `action=retrain`
trigger a run; Airflow rejects a second concurrent run (max_active_runs=1) so a flapping alert
cannot stack up trainings.
"""

from __future__ import annotations

import logging
import os

import httpx
from fastapi import FastAPI, HTTPException

log = logging.getLogger(__name__)
DAG_ID = "retrain"


def trigger_retrain(
    client: httpx.Client, base_url: str, user: str, password: str, reason: str
) -> str:
    """Trigger the DAG. Airflow 3 uses JWT (`/auth/token` + `/api/v2`), Airflow 2 uses basic auth
    on `/api/v1`; the bridge tries v3 first and falls back."""
    token_resp = client.post(
        f"{base_url}/auth/token", json={"username": user, "password": password}
    )
    if token_resp.status_code in (200, 201):
        token = token_resp.json()["access_token"]
        r = client.post(
            f"{base_url}/api/v2/dags/{DAG_ID}/dagRuns",
            headers={"Authorization": f"Bearer {token}"},
            json={"logical_date": None, "conf": {"reason": reason}},
        )
    else:
        r = client.post(
            f"{base_url}/api/v1/dags/{DAG_ID}/dagRuns",
            auth=(user, password),
            json={"conf": {"reason": reason}},
        )
    r.raise_for_status()
    return str(r.json().get("dag_run_id", "unknown"))


def retrain_reasons(payload: dict) -> list[str]:
    """Names of firing alerts that ask for a retrain."""
    return [
        a["labels"].get("alertname", "unknown")
        for a in payload.get("alerts", [])
        if a.get("status") == "firing" and a.get("labels", {}).get("action") == "retrain"
    ]


def create_app(client: httpx.Client | None = None) -> FastAPI:
    app = FastAPI(title="DemandOps alert bridge")
    base_url = os.environ.get("AIRFLOW_URL", "http://airflow-api-server.mlops.svc:8080")
    user = os.environ.get("AIRFLOW_USER", "admin")
    password = os.environ.get("AIRFLOW_PASSWORD", "admin")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/alert")
    def alert(payload: dict) -> dict[str, object]:
        reasons = retrain_reasons(payload)
        if not reasons:
            return {"triggered": False}
        http = client or httpx.Client(timeout=30)
        try:
            run_id = trigger_retrain(http, base_url, user, password, ",".join(sorted(set(reasons))))
        except httpx.HTTPError as exc:
            log.error("could not trigger retrain: %s", exc)
            raise HTTPException(502, "airflow unreachable or rejected the request") from exc
        log.warning("retrain triggered by %s -> %s", reasons, run_id)
        return {"triggered": True, "dag_run_id": run_id, "reasons": reasons}

    return app


def build_app() -> FastAPI:  # pragma: no cover
    return create_app()
