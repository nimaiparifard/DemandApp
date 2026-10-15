import httpx
from fastapi.testclient import TestClient

from demandops.monitoring.alert_bridge import create_app, retrain_reasons

FIRING = {
    "alerts": [
        {"status": "firing", "labels": {"alertname": "DemandDataDrift", "action": "retrain"}},
        {"status": "firing", "labels": {"alertname": "ServingHighLatencyP95"}},
    ]
}


def _airflow(calls, v3=True):
    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path, req.headers.get("authorization", "")[:6]))
        if req.url.path == "/auth/token":
            return httpx.Response(201, json={"access_token": "jwt"}) if v3 else httpx.Response(404)
        if req.url.path == "/api/v2/dags/retrain/dagRuns":
            assert req.headers["authorization"] == "Bearer jwt"
            return httpx.Response(200, json={"dag_run_id": "manual__1"})
        if req.url.path == "/api/v1/dags/retrain/dagRuns":
            return httpx.Response(200, json={"dag_run_id": "manual__2"})
        return httpx.Response(500)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_only_firing_retrain_alerts_count():
    assert retrain_reasons(FIRING) == ["DemandDataDrift"]
    assert (
        retrain_reasons({"alerts": [{"status": "resolved", "labels": {"action": "retrain"}}]}) == []
    )
    assert retrain_reasons({}) == []


def test_firing_alert_triggers_dag_with_airflow3_jwt():
    calls: list = []
    r = TestClient(create_app(_airflow(calls))).post("/alert", json=FIRING)
    assert r.json() == {
        "triggered": True,
        "dag_run_id": "manual__1",
        "reasons": ["DemandDataDrift"],
    }
    assert [c[1] for c in calls] == ["/auth/token", "/api/v2/dags/retrain/dagRuns"]


def test_falls_back_to_airflow2_basic_auth():
    calls: list = []
    r = TestClient(create_app(_airflow(calls, v3=False))).post("/alert", json=FIRING)
    assert r.json()["dag_run_id"] == "manual__2"
    assert calls[-1][1] == "/api/v1/dags/retrain/dagRuns"


def test_non_retrain_alert_does_not_call_airflow():
    calls: list = []
    payload = {"alerts": [{"status": "firing", "labels": {"alertname": "ServingErrorRate"}}]}
    assert TestClient(create_app(_airflow(calls))).post("/alert", json=payload).json() == {
        "triggered": False
    }
    assert calls == []


def test_airflow_rejection_returns_502_so_alertmanager_retries():
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503, json={})))
    assert TestClient(create_app(client)).post("/alert", json=FIRING).status_code == 502
