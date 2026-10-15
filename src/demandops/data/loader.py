"""Load M5 CSVs into a compact long-format panel.

Wide -> long melt of the full data set is ~59M rows, so loading supports a
deterministic sample of series (default 100) for development and CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from demandops.data.schemas import ID_COLS

CAL_COLS = [
    "d", "date", "wm_yr_wk", "wday", "month", "year",
    "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI",
]  # fmt: skip


def read_calendar(data_dir: Path) -> pd.DataFrame:
    cal = pd.read_csv(data_dir / "calendar.csv", usecols=CAL_COLS)
    cal["date"] = pd.to_datetime(cal["date"])
    cal["d"] = cal["d"].str.slice(2).astype(int)
    return cal


def trim_history(
    sales: pd.DataFrame, history_days: int, last_day: int | None = None
) -> pd.DataFrame:
    """Keep only the id columns and the `history_days` most recent day columns (<= last_day)."""
    if not history_days:
        return sales
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    last = last_day if last_day is not None else max(int(c[2:]) for c in day_cols)
    keep = [c for c in day_cols if last - history_days < int(c[2:]) <= last]
    id_cols = [c for c in sales.columns if not c.startswith("d_")]
    return sales[[*id_cols, *keep]]


def read_sales_wide(
    data_dir: Path, sales_file: str, n_series: int | None, seed: int = 0
) -> pd.DataFrame:
    """Read the wide sales table, keeping a deterministic sample of series."""
    sales = pd.read_csv(data_dir / sales_file)
    if n_series and n_series < len(sales):
        sales = sales.sample(n=n_series, random_state=seed).sort_values("id").reset_index(drop=True)
    return sales


def wide_to_long(sales: pd.DataFrame) -> pd.DataFrame:
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    days = np.array([int(c[2:]) for c in day_cols], dtype=np.int32)
    n, t = len(sales), len(day_cols)
    values = sales[day_cols].to_numpy(dtype=np.float32)
    long = pd.DataFrame(
        {
            "id": np.repeat(sales["id"].to_numpy(), t),
            "d": np.tile(days, n),
            "sales": values.ravel(),
        }
    )
    ids = sales[ID_COLS].drop_duplicates("id")
    return long.merge(ids, on="id", how="left")


def read_prices(
    data_dir: Path, items: set[str] | None = None, stores: set[str] | None = None
) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(data_dir / "sell_prices.csv", chunksize=2_000_000):
        if items is not None:
            chunk = chunk[chunk["item_id"].isin(items)]
        if stores is not None:
            chunk = chunk[chunk["store_id"].isin(stores)]
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def attach_context(
    long: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Join calendar + price onto a long (id, d, sales, ids...) frame; sorted by (id, d)."""
    long = long.drop(columns=["date"], errors="ignore")
    panel = long.merge(calendar, on="d", how="left")
    panel = panel.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    panel["sell_price"] = panel["sell_price"].astype("float32")
    return panel.sort_values(["id", "d"]).reset_index(drop=True)


def build_panel(
    sales_wide: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame
) -> pd.DataFrame:
    """Join sales + calendar + price into one sorted long panel."""
    return attach_context(wide_to_long(sales_wide), calendar, prices)


def load_panel(
    data_dir: Path,
    sales_file: str,
    n_series: int | None,
    seed: int = 0,
    history_days: int = 0,
) -> pd.DataFrame:
    sales = trim_history(read_sales_wide(data_dir, sales_file, n_series, seed), history_days)
    cal = read_calendar(data_dir)
    prices = read_prices(data_dir, set(sales["item_id"]), set(sales["store_id"]))
    return build_panel(sales, cal, prices)


def extend_horizon(panel: pd.DataFrame, calendar: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Append `horizon` future rows (sales = NaN) per series.

    The future calendar is known (M5 calendar covers d_1..d_1969); future prices
    are unknown, so the last observed price is carried forward.
    """
    last_d = int(panel["d"].max())
    future_cal = calendar[(calendar["d"] > last_d) & (calendar["d"] <= last_d + horizon)]
    if len(future_cal) < horizon:
        raise ValueError(
            f"calendar covers only {len(future_cal)} of {horizon} future days after d_{last_d}"
        )
    meta = panel.drop_duplicates("id")[ID_COLS]
    future = meta.merge(future_cal, how="cross")
    future["sales"] = np.float32(np.nan)
    last_price = (
        panel.dropna(subset=["sell_price"]).sort_values("d").groupby("id")["sell_price"].last()
    )
    future["sell_price"] = future["id"].map(last_price).astype("float32")
    out = pd.concat([panel, future], ignore_index=True)
    return out.sort_values(["id", "d"]).reset_index(drop=True)
