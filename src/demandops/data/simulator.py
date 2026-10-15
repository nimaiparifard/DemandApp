"""Data simulator: replays the M5 history as if it were live.

Each `tick` releases ONE new day of sales (all series) into the landing bucket as
`sales_YYYY-MM-DD.parquet`. A tiny `_state.json` remembers the replay position, so
the process is restartable (it runs as a Kubernetes CronJob every N minutes).

Drift can be injected on demand (`demand_shift`) to demo the drift -> retrain loop.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from demandops.data.loader import read_calendar, wide_to_long
from demandops.storage import ObjectStore

STATE_FILE = "_state.json"


class Simulator:
    def __init__(
        self,
        store: ObjectStore,
        data_dir: Path,
        sales_file: str,
        start_day: int = 1913,
        n_series: int | None = None,
        seed: int = 0,
    ) -> None:
        self.store = store
        self.start_day = start_day
        sales = pd.read_csv(data_dir / sales_file)
        if n_series and n_series < len(sales):
            sales = (
                sales.sample(n=n_series, random_state=seed).sort_values("id").reset_index(drop=True)
            )
        self.sales = sales
        self.calendar = read_calendar(data_dir)
        self.last_day = int(max(int(c[2:]) for c in sales.columns if c.startswith("d_")))

    # ---- state -------------------------------------------------------
    def position(self) -> int:
        """Last day already released."""
        if self.store.exists(STATE_FILE):
            return int(self.store.read_json(STATE_FILE)["last_released"])
        return self.start_day

    def _day_frame(self, d: int, demand_shift: float) -> pd.DataFrame:
        col = f"d_{d}"
        frame = self.sales[["id", col]].rename(columns={col: "sales"})
        frame["d"] = d
        frame["date"] = self.calendar.loc[self.calendar["d"] == d, "date"].iloc[0]
        frame["sales"] = frame["sales"].astype("float32")
        if demand_shift != 1.0:  # synthetic drift for demos
            rng = np.random.default_rng(d)
            frame["sales"] = rng.poisson(frame["sales"].to_numpy() * demand_shift).astype("float32")
        return frame

    def tick(self, demand_shift: float = 1.0) -> str | None:
        """Release the next day. Returns the object name, or None when history is exhausted."""
        nxt = self.position() + 1
        if nxt > self.last_day:
            return None
        frame = self._day_frame(nxt, demand_shift)
        name = f"sales_{frame['date'].iloc[0]:%Y-%m-%d}.parquet"
        self.store.write_parquet(frame, name)
        self.store.write_json({"last_released": nxt}, STATE_FILE)
        return name

    def catch_up(self, n_days: int, demand_shift: float = 1.0) -> list[str]:
        released = []
        for _ in range(n_days):
            name = self.tick(demand_shift)
            if name is None:
                break
            released.append(name)
        return released


def read_landed(landing: ObjectStore) -> pd.DataFrame | None:
    """All days released by the simulator (None if nothing landed yet)."""
    files = [f for f in landing.list() if f.startswith("sales_") and f.endswith(".parquet")]
    if not files:
        return None
    landed = pd.concat([landing.read_parquet(f) for f in files], ignore_index=True)
    return landed.drop(columns=["date"]).drop_duplicates(["id", "d"], keep="last")


def landing_to_long(
    sales_wide: pd.DataFrame,
    landing: ObjectStore,
    start_day: int,
    landed: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Base history (d <= start_day) + every landed day -> long sales frame (no calendar yet).

    Pass `landed` (from `read_landed`) to avoid re-reading the landing bucket when this is
    called once per chunk of series."""
    keep = [c for c in sales_wide.columns if not c.startswith("d_") or int(c[2:]) <= start_day]
    base = wide_to_long(sales_wide[keep])
    if landed is None:
        landed = read_landed(landing)
    if landed is None:
        return base
    landed = landed[landed["id"].isin(sales_wide["id"])]
    meta = sales_wide[["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]]
    landed = landed.merge(meta, on="id", how="left")
    out = pd.concat(
        [base[base["d"] <= start_day], landed[landed["d"] > start_day]], ignore_index=True
    )
    return out.sort_values(["id", "d"]).reset_index(drop=True)
