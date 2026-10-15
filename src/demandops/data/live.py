"""Assemble the 'current' panel: base history + days released by the simulator."""

from __future__ import annotations

import pandas as pd

from demandops.config import Settings
from demandops.data.loader import (
    attach_context,
    read_calendar,
    read_prices,
    read_sales_wide,
    trim_history,
)
from demandops.data.simulator import landing_to_long, read_landed
from demandops.features.store import horizon_feature_table
from demandops.storage import ObjectStore


def landing_store(settings: Settings) -> ObjectStore:
    return ObjectStore(settings.landing_uri, settings.s3_endpoint_url)


def load_live_panel(
    settings: Settings, start_day: int | None = None, history_days: int | None = None
) -> pd.DataFrame:
    """history_days overrides settings.history_days (the ingest/monitoring jobs pass the shorter
    `ingest_history_days`)."""
    start_day = settings.simulator_start_day if start_day is None else start_day
    history_days = settings.history_days if history_days is None else history_days
    wide = read_sales_wide(
        settings.demandops_data_dir,
        settings.sales_file,
        settings.demandops_n_series,
        settings.train.seed,
    )
    wide = trim_history(wide, history_days, last_day=start_day)
    long = landing_to_long(wide, landing_store(settings), start_day)
    cal = read_calendar(settings.demandops_data_dir)
    prices = read_prices(settings.demandops_data_dir, set(wide["item_id"]), set(wide["store_id"]))
    return attach_context(long, cal, prices)


def build_horizon_table(settings: Settings, chunk_series: int = 5000) -> pd.DataFrame:
    """Feature table for the next 28 days of every series, built chunk by chunk.

    Series are independent, so processing `chunk_series` at a time gives exactly the same
    result as one big frame while bounding memory (all 30,490 series x 450 days at once
    needs ~10 GB; a 5,000-series chunk needs ~1.5 GB).
    """
    start_day = settings.simulator_start_day
    wide_all = read_sales_wide(
        settings.demandops_data_dir,
        settings.sales_file,
        settings.demandops_n_series,
        settings.train.seed,
    )
    wide_all = trim_history(wide_all, settings.ingest_history_days, last_day=start_day)
    cal = read_calendar(settings.demandops_data_dir)
    prices = read_prices(
        settings.demandops_data_dir, set(wide_all["item_id"]), set(wide_all["store_id"])
    )
    landed = read_landed(landing_store(settings))
    parts = []
    for lo in range(0, len(wide_all), chunk_series):
        wide = wide_all.iloc[lo : lo + chunk_series]
        long = landing_to_long(wide, landing_store(settings), start_day, landed=landed)
        chunk_prices = prices[prices["item_id"].isin(set(wide["item_id"]))]
        panel = attach_context(long, cal, chunk_prices)
        parts.append(horizon_feature_table(panel, cal, settings.features))
    table = pd.concat(parts, ignore_index=True)
    return table.sort_values(["id", "d"]).reset_index(drop=True)
