"""Generate a small M5-shaped data set (calendar.csv, sales_train_evaluation.csv, sell_prices.csv).

For CI and for trying the platform WITHOUT the Kaggle download. The numbers are synthetic
(weekly seasonality + trend + Poisson noise), so accuracy results from it mean nothing: use the real
M5 files for the results table.

    python scripts/make_synthetic_m5.py --out data_synth --series 60 --days 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

STATES = {"CA": ["CA_1", "CA_2"], "TX": ["TX_1"], "WI": ["WI_1"]}
DEPTS = [
    ("FOODS_1", "FOODS"),
    ("FOODS_2", "FOODS"),
    ("HOBBIES_1", "HOBBIES"),
    ("HOUSEHOLD_1", "HOUSEHOLD"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data_synth"))
    ap.add_argument("--series", type=int, default=60)
    ap.add_argument("--days", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    total_days = a.days + 28  # calendar extends past the data so the horizon is known
    dates = pd.date_range("2014-01-01", periods=total_days)
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11300 + np.arange(total_days) // 7,
            "weekday": dates.day_name(),
            "wday": dates.dayofweek + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i + 1}" for i in range(total_days)],
            "event_name_1": np.where(dates.day == 25, "Holiday", None),
            "event_type_1": np.where(dates.day == 25, "National", None),
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": rng.integers(0, 2, total_days),
            "snap_TX": rng.integers(0, 2, total_days),
            "snap_WI": rng.integers(0, 2, total_days),
        }
    )
    cal.to_csv(a.out / "calendar.csv", index=False)

    stores = [(s, st) for st, ss in STATES.items() for s in ss]
    rows, price_rows = [], []
    weekly = np.array([0.9, 0.85, 0.9, 1.0, 1.2, 1.6, 1.5])
    n_items = max(1, a.series // len(stores))
    for i in range(n_items):
        dept, cat = DEPTS[i % len(DEPTS)]
        item = f"{dept}_{i + 1:03d}"
        for store, state in stores:
            base = rng.uniform(0.5, 7)
            trend = rng.uniform(-0.0005, 0.002)
            lam = base * weekly[np.arange(total_days) % 7] * (1 + trend * np.arange(total_days))
            sales = rng.poisson(np.maximum(lam, 0.05))[: a.days]
            rows.append(
                {"id": f"{item}_{store}_evaluation", "item_id": item, "dept_id": dept, "cat_id": cat,
                 "store_id": store, "state_id": state, **{f"d_{d + 1}": int(sales[d]) for d in range(a.days)}}
            )  # fmt: skip
            p0 = rng.uniform(1.5, 12)
            for wk in np.unique(cal["wm_yr_wk"]):
                price_rows.append(
                    (store, item, int(wk), round(p0 * (1 + 0.03 * rng.standard_normal()), 2))
                )
    pd.DataFrame(rows).to_csv(a.out / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(price_rows, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        a.out / "sell_prices.csv", index=False
    )
    print(f"wrote {len(rows)} series x {a.days} days to {a.out}/")


if __name__ == "__main__":
    main()
