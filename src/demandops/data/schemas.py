"""pandera schemas for the long-format panel that every downstream step consumes."""

from __future__ import annotations

import pandera.pandas as pa

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Long panel: one row per (series, day).
PanelSchema = pa.DataFrameSchema(
    {
        "id": pa.Column(str),
        "d": pa.Column(int, pa.Check.ge(1), coerce=True),
        "date": pa.Column("datetime64[ns]"),
        # NaN sales = future (horizon) rows; NaN price = item not on sale yet
        "sales": pa.Column(float, pa.Check.ge(0), nullable=True, coerce=True),
        "sell_price": pa.Column(float, pa.Check.gt(0), nullable=True, coerce=True),
    },
    strict=False,
)
