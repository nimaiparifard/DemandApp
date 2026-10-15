"""Feast definitions (entity, source, feature view).

Offline store: parquet on the S3 store (s3://demandops-features/horizon_features.parquet).
Online store: Redis (configured in feature_store.yaml).
Materialisation is run by DAG 1 (`daily_ingest`) after validation.

NOTE: this module needs `pip install feast[redis]`; it is excluded from the unit-test run.
"""

from __future__ import annotations

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int32, String

fkey = Entity(name="fkey", join_keys=["fkey"], description="'<series id>|<target day number>'")

horizon_source = FileSource(
    name="horizon_features_source",
    path="s3://demandops-features/horizon_features.parquet",
    timestamp_field="event_timestamp",
    s3_endpoint_override="http://s3.storage.svc:8333",
)

_f32 = [
    "lag_28", "lag_35", "lag_42",
    "rmean_7", "rstd_7", "rmean_28", "rstd_28", "rmean_56", "rstd_56",
    "sell_price", "price_change_wk", "price_vs_item_mean",
    "price_last_week", "price_item_mean",
]  # fmt: skip
_i32 = ["dow", "month", "dom", "has_event", "snap"]
_str = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_type_1"]

horizon_features = FeatureView(
    name="horizon_features",
    entities=[fkey],
    ttl=timedelta(days=3),
    schema=[
        *(Field(name=n, dtype=Float32) for n in _f32),
        *(Field(name=n, dtype=Int32) for n in _i32),
        *(Field(name=n, dtype=String) for n in _str),
    ],
    source=horizon_source,
)
