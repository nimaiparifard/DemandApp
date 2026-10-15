"""Wire the real dependencies: model from the MLflow registry + features from Feast (or parquet)."""

from __future__ import annotations

import os
from collections.abc import Callable

from demandops.config import Settings
from demandops.serving.service import (
    FeastFeatureProvider,
    FeatureProvider,
    ForecastService,
    ParquetFeatureProvider,
)
from demandops.storage import ObjectStore
from demandops.training.train import load_model


def default_service_factory(settings: Settings) -> Callable[[], ForecastService]:
    """MODEL_VERSION (pinned by GitOps) wins over MODEL_ALIAS (default 'champion')."""

    def factory() -> ForecastService:
        version = os.environ.get("MODEL_VERSION") or None
        model, resolved = load_model(
            settings, version=version, alias=None if version else settings.model_alias
        )
        provider: FeatureProvider
        if os.environ.get("FEATURE_BACKEND", "feast") == "feast":
            first_day = (
                int(os.environ["FIRST_TARGET_DAY"])
                if "FIRST_TARGET_DAY" in os.environ
                else _first_target_day(settings)
            )
            provider = FeastFeatureProvider(
                os.environ.get("FEAST_REPO", "/app/feature_repo"), first_day
            )
        else:  # parquet fallback (local docker-compose / demo without Feast)
            table = ObjectStore(settings.features_uri, settings.s3_endpoint_url).read_parquet(
                "horizon_features.parquet"
            )
            provider = ParquetFeatureProvider(table)
        return ForecastService(model, resolved, provider)

    return factory


def _first_target_day(settings: Settings) -> int:
    table = ObjectStore(settings.features_uri, settings.s3_endpoint_url).read_parquet(
        "horizon_features.parquet"
    )
    return int(table["d"].min())
