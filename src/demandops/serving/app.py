"""Inference API. Container image used by the KServe InferenceService (custom container)
and runnable on its own: `uvicorn demandops.serving.app:app`.

  POST /v2/models/demand-forecaster/infer   {item_id, store_id, horizon<=28, price_override?}
  GET  /forecasts/{store_id}?date=YYYY-MM-DD   (batch table in Postgres)
  GET  /health   /v2/health/ready   /metrics
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from demandops.serving.service import ForecastService, SeriesNotFound

MODEL_ROUTE = "/v2/models/demand-forecaster/infer"


class InferRequest(BaseModel):
    item_id: str
    store_id: str
    horizon: int = Field(28, ge=1, le=28)
    price_override: float | None = Field(None, gt=0)


class DayForecast(BaseModel):
    date: date
    forecast: float
    lower: float
    upper: float


class InferResponse(BaseModel):
    model_name: str = "demand-forecaster"
    model_version: str
    item_id: str
    store_id: str
    forecasts: list[DayForecast]


def create_app(
    service_factory: Callable[[], ForecastService],
    forecast_reader: Callable[[str, date | None], list[dict]] | None = None,
    registry: CollectorRegistry | None = None,
) -> FastAPI:
    """service_factory is called lazily at startup (and on /admin/reload), so importing this
    module never touches MLflow or Redis."""
    registry = registry or CollectorRegistry()
    req_count = Counter(
        "demand_requests_total", "Requests", ["route", "status", "model_version"], registry=registry
    )
    req_latency = Histogram(
        "demand_request_seconds", "Request latency", ["route", "model_version"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5), registry=registry,
    )  # fmt: skip
    pred_mean = Histogram(
        "demand_prediction_units", "Distribution of daily point forecasts",
        buckets=(0, 0.5, 1, 2, 5, 10, 25, 50, 100), registry=registry,
    )  # fmt: skip
    model_info = Gauge(
        "demand_model_info", "Loaded model version", ["model_version"], registry=registry
    )

    state: dict[str, ForecastService | None] = {"service": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        load()
        yield

    app = FastAPI(title="DemandOps inference", version="0.1.0", lifespan=lifespan)

    def load() -> None:
        svc = service_factory()
        state["service"] = svc
        model_info.clear()
        model_info.labels(model_version=svc.version).set(1)

    def service() -> ForecastService:
        svc = state["service"]
        if svc is None:
            load()
            svc = state["service"]
        assert svc is not None
        return svc

    @app.middleware("http")
    async def _metrics(request: Request, call_next):  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)
        svc = state["service"]
        version = svc.version if svc else "none"
        req_count.labels(path, str(response.status_code), version).inc()
        req_latency.labels(path, version).observe(time.perf_counter() - t0)
        return response

    @app.get("/health")
    @app.get("/v2/health/ready")
    def health() -> dict[str, str]:
        svc = state["service"]
        if svc is None:
            raise HTTPException(503, "model not loaded")
        return {"status": "ok", "model_version": svc.version}

    @app.post(MODEL_ROUTE, response_model=InferResponse)
    def infer(body: InferRequest) -> InferResponse:
        svc = service()
        try:
            fc = svc.forecast(body.item_id, body.store_id, body.horizon, body.price_override)
        except SeriesNotFound as exc:
            raise HTTPException(404, f"unknown series {exc}") from exc
        dates = _day_to_date(fc["d"].tolist(), svc)
        for p in fc["pred"]:
            pred_mean.observe(float(p))
        return InferResponse(
            model_version=svc.version,
            item_id=body.item_id,
            store_id=body.store_id,
            forecasts=[
                DayForecast(
                    date=dt,
                    forecast=round(float(p), 4),
                    lower=round(float(lo), 4),
                    upper=round(float(hi), 4),
                )
                for dt, p, lo, hi in zip(dates, fc["pred"], fc["lower"], fc["upper"], strict=True)
            ],
        )

    @app.get("/forecasts/{store_id}")
    def forecasts(store_id: str, date: date | None = None) -> list[dict]:
        if forecast_reader is None:
            raise HTTPException(501, "batch forecast table not configured")
        return forecast_reader(store_id, date)

    @app.post("/admin/reload")
    def reload() -> dict[str, str]:
        load()
        return {"model_version": service().version}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


def _day_to_date(days: list[int], svc: ForecastService) -> list[date]:
    """Day number -> calendar date. The lookup is built once per loaded service, not per request."""
    lookup = getattr(svc, "_date_lookup", None)
    if lookup is None:
        table = getattr(svc.provider, "table", None)
        if table is not None:
            lookup = {int(d): ts.date() for d, ts in zip(table["d"], table["date"], strict=True)}
        else:
            from demandops.config import get_settings
            from demandops.data.loader import read_calendar

            cal = read_calendar(get_settings().demandops_data_dir)
            lookup = {int(d): ts.date() for d, ts in zip(cal["d"], cal["date"], strict=True)}
        svc._date_lookup = lookup  # type: ignore[attr-defined]
    return [lookup[d] for d in days]


def build_default_app() -> FastAPI:  # pragma: no cover - needs MLflow/Feast
    from demandops.batch.forecast_table import make_reader
    from demandops.config import get_settings
    from demandops.serving.factory import default_service_factory

    settings = get_settings()
    reader = (
        make_reader(settings.database_url)
        if os.environ.get("ENABLE_FORECAST_TABLE", "1") == "1"
        else None
    )
    return create_app(default_service_factory(settings), reader)


def __getattr__(name: str) -> FastAPI:  # `uvicorn demandops.serving.app:app` builds lazily
    if name == "app":
        return build_default_app()
    raise AttributeError(name)
