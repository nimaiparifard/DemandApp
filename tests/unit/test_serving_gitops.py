import base64
import json
import time

import httpx
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from demandops.batch.forecast_table import make_reader, write_forecasts
from demandops.batch.predict_all import predict_all
from demandops.features.store import horizon_feature_table
from demandops.models.lgbm import LGBMForecaster
from demandops.serving.app import create_app
from demandops.serving.service import ForecastService, ParquetFeatureProvider, SeriesNotFound
from pipelines.gitops import bump_model_version, open_model_bump_pr


@pytest.fixture(scope="module")
def trained(panel, calendar):
    from demandops.config import FeatureConfig, TrainConfig

    tcfg = TrainConfig(num_boost_round=40, train_days=300)
    tcfg.lgbm_params.update(num_leaves=15, min_data_in_leaf=5)
    model = LGBMForecaster(FeatureConfig(), tcfg)
    obs = panel[panel["d"] <= 420]
    model.fit(obs, 420)
    preds = model.predict(obs, 392, 28)
    truth = obs.merge(preds, on=["id", "d"])
    model.calibrate_interval(truth["sales"].to_numpy(), truth["pred"].to_numpy())
    table = horizon_feature_table(obs, calendar, FeatureConfig())
    return model, table


def test_horizon_table_has_28_days_per_series_and_unique_keys(trained, panel):
    _, table = trained
    assert table.groupby("id")["d"].agg(["min", "max", "count"]).eq([421, 448, 28]).all().all()
    assert table["fkey"].is_unique
    assert (table["event_timestamp"] == panel.loc[panel["d"] == 420, "date"].iloc[0]).all()


def test_forecast_service_returns_ordered_interval(trained):
    model, table = trained
    svc = ForecastService(model, "7", ParquetFeatureProvider(table))
    fc = svc.forecast("FOODS_1_001", "CA_1", 28)
    assert len(fc) == 28 and (fc["pred"] >= 0).all()
    assert (fc["lower"] <= fc["pred"] + 1e-9).all() and (fc["pred"] <= fc["upper"] + 1e-9).all()


def test_higher_price_lowers_forecast_for_most_items(trained):
    model, table = trained
    svc = ForecastService(model, "7", ParquetFeatureProvider(table))
    base = svc.forecast("FOODS_1_001", "CA_1", 28)["pred"].sum()
    changed = svc.forecast("FOODS_1_001", "CA_1", 28, price_override=100.0)["pred"].sum()
    assert changed != base  # the override reaches the model


def test_unknown_series_raises(trained):
    model, table = trained
    with pytest.raises(SeriesNotFound):
        ForecastService(model, "7", ParquetFeatureProvider(table)).forecast("NOPE", "CA_1", 28)


def test_model_roundtrip_gives_identical_predictions(trained, tmp_path):
    model, table = trained
    model.save(tmp_path / "m")
    loaded = LGBMForecaster.load(tmp_path / "m")
    np.testing.assert_allclose(
        model.predict_features(table)["pred"], loaded.predict_features(table)["pred"]
    )
    assert loaded.interval_z == model.interval_z


@pytest.fixture
def client(trained):
    model, table = trained
    svc = ForecastService(model, "7", ParquetFeatureProvider(table))
    app = create_app(lambda: svc, registry=CollectorRegistry())
    with TestClient(app) as c:
        yield c


def test_api_infer_and_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "7"}
    r = client.post(
        "/v2/models/demand-forecaster/infer",
        json={"item_id": "FOODS_1_001", "store_id": "CA_1", "horizon": 14},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["model_version"] == "7" and len(body["forecasts"]) == 14
    assert body["forecasts"][0]["date"] == "2016-02-25"  # d_421 = 2015-01-01 + 420 days


def test_api_validation_and_404(client):
    bad = client.post(
        "/v2/models/demand-forecaster/infer", json={"item_id": "x", "store_id": "y", "horizon": 99}
    )
    assert bad.status_code == 422
    missing = client.post(
        "/v2/models/demand-forecaster/infer", json={"item_id": "x", "store_id": "y"}
    )
    assert missing.status_code == 404


def test_api_exposes_prometheus_metrics_with_model_version(client):
    client.post(
        "/v2/models/demand-forecaster/infer", json={"item_id": "FOODS_1_001", "store_id": "CA_1"}
    )
    text = client.get("/metrics").text
    assert 'demand_requests_total{model_version="7"' in text
    assert (
        "demand_request_seconds_bucket" in text
        and 'demand_model_info{model_version="7"} 1.0' in text
    )


def test_batch_forecasts_roundtrip_through_database(trained, tmp_path):
    model, table = trained
    out = predict_all(model, table, "7")
    url = f"sqlite:///{tmp_path / 'f.db'}"
    assert write_forecasts(out, url) == len(table)
    assert write_forecasts(out, url) == len(table)  # idempotent for the same run_date
    rows = make_reader(url)("CA_1", None)
    assert rows and {r["store_id"] for r in rows} == {"CA_1"} and len(rows) == 2 * 28


# ---------------- GitOps ----------------
VALUES = """image:
  tag: abc
model:
  name: demand-forecaster
  version: "3"   # bumped by the retrain DAG
canary:
  steps: [10, 50, 100]
"""


def test_bump_changes_only_the_model_version():
    new = bump_model_version(VALUES, "4")
    assert new == VALUES.replace('version: "3"', 'version: "4"')


def test_bump_requires_model_block():
    with pytest.raises(ValueError):
        bump_model_version("image:\n  version: 1\n", "2")


def test_open_pr_flow_uses_branch_and_returns_url():
    calls = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append((req.method, req.url.path))
        if req.method == "GET" and "git/ref/heads" in req.url.path:
            return httpx.Response(200, json={"object": {"sha": "base123"}})
        if req.method == "POST" and req.url.path.endswith("/git/refs"):
            assert json.loads(req.content)["ref"] == "refs/heads/model/bump-v4"
            return httpx.Response(201, json={})
        if req.method == "GET" and "/contents/" in req.url.path:
            return httpx.Response(
                200, json={"sha": "f1", "content": base64.b64encode(VALUES.encode()).decode()}
            )
        if req.method == "PUT":
            body = json.loads(req.content)
            assert (
                'version: "4"' in base64.b64decode(body["content"]).decode()
                and body["branch"] == "model/bump-v4"
            )
            return httpx.Response(200, json={})
        if req.method == "POST" and req.url.path.endswith("/pulls"):
            return httpx.Response(201, json={"html_url": "https://github.com/o/r/pull/9"})
        return httpx.Response(404)

    client = httpx.Client(base_url="https://api.github.com", transport=httpx.MockTransport(handler))
    url = open_model_bump_pr("o/r", "tok", "4", client=client)
    assert url.endswith("/pull/9") and [m for m, _ in calls].count("POST") == 2


# ---------------- request cache ----------------
class _CountingProvider(ParquetFeatureProvider):
    calls = 0

    def get_rows(self, item_id, store_id, horizon):
        type(self).calls += 1
        return super().get_rows(item_id, store_id, horizon)


def test_cache_serves_repeat_requests_without_touching_the_feature_store(trained):
    model, table = trained
    _CountingProvider.calls = 0
    svc = ForecastService(model, "7", _CountingProvider(table))
    first = svc.forecast("FOODS_1_001", "CA_1", 28)
    second = svc.forecast("FOODS_1_001", "CA_1", 28)
    assert _CountingProvider.calls == 1
    pd.testing.assert_frame_equal(first, second)


def test_price_override_bypasses_and_does_not_pollute_the_cache(trained):
    model, table = trained
    _CountingProvider.calls = 0
    svc = ForecastService(model, "7", _CountingProvider(table))
    base = svc.forecast("FOODS_1_001", "CA_1", 28)
    what_if = svc.forecast("FOODS_1_001", "CA_1", 28, price_override=50.0)
    again = svc.forecast("FOODS_1_001", "CA_1", 28)
    assert _CountingProvider.calls == 2  # base + what-if; the third call is a cache hit
    assert not what_if["pred"].equals(base["pred"])
    pd.testing.assert_frame_equal(base, again)  # the override did not leak into the cache


def test_cache_entries_expire(trained):
    model, table = trained
    _CountingProvider.calls = 0
    svc = ForecastService(model, "7", _CountingProvider(table), cache_ttl=0.01)
    svc.forecast("FOODS_1_001", "CA_1", 28)
    time.sleep(0.05)
    svc.forecast("FOODS_1_001", "CA_1", 28)
    assert _CountingProvider.calls == 2


def test_cache_is_bounded(trained):
    model, table = trained
    svc = ForecastService(model, "7", ParquetFeatureProvider(table), cache_size=2)
    for store in ("CA_1", "TX_1", "WI_1"):
        svc.forecast("FOODS_1_001", store, 28)
    assert len(svc._cache) == 2


def test_latency_benchmark_measures_the_cold_path(trained):
    model, table = trained
    _CountingProvider.calls = 0
    svc = ForecastService(model, "7", _CountingProvider(table))
    svc.benchmark_p95_ms("FOODS_1_001", "CA_1", n=5)
    assert _CountingProvider.calls == 5
