"""Forecast table (Postgres in the cluster, SQLite locally) - same SQLAlchemy code for both."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text

TABLE = "forecasts"


def write_forecasts(df: pd.DataFrame, database_url: str) -> int:
    """df: id, store_id, item_id, date, forecast, lower, upper, model_version, run_date.
    Replaces the rows of the same run_date so the DAG is idempotent."""
    engine = create_engine(database_url)
    run_date = str(df["run_date"].iloc[0])
    with engine.begin() as conn:
        if engine.dialect.has_table(conn, TABLE):
            conn.execute(text(f"DELETE FROM {TABLE} WHERE run_date = :r"), {"r": run_date})
        df.to_sql(TABLE, conn, if_exists="append", index=False, chunksize=10_000)
        conn.execute(
            text(f"CREATE INDEX IF NOT EXISTS ix_{TABLE}_store_date ON {TABLE} (store_id, date)")
        )
    return len(df)


def make_reader(database_url: str) -> Callable[[str, date | None], list[dict]]:
    engine = create_engine(database_url)

    def read(store_id: str, on: date | None) -> list[dict]:
        sql = (
            f"SELECT id, item_id, store_id, date, forecast, lower, upper, model_version FROM {TABLE} "
            f"WHERE store_id = :s AND run_date = (SELECT MAX(run_date) FROM {TABLE})"
        )
        params: dict[str, str] = {"s": store_id}
        if on is not None:
            sql += " AND date = :d"
            params["d"] = on.isoformat()
        sql += " ORDER BY item_id, date LIMIT 5000"
        with engine.connect() as conn:
            return [dict(r._mapping) for r in conn.execute(text(sql), params)]

    return read
