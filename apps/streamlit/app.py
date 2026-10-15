"""Forecast viewer + what-if price simulator (talks to the inference API)."""

from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("API_URL", "http://localhost:8001")

st.set_page_config(page_title="DemandOps", page_icon=":material/inventory_2:", layout="wide")
st.title("DemandOps - 28-day demand forecast")


@st.cache_data(ttl=60, show_spinner=False)
def fetch_store(store_id: str) -> pd.DataFrame:
    r = httpx.get(f"{API}/forecasts/{store_id}", timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def infer(item_id: str, store_id: str, price: float | None) -> pd.DataFrame:
    body = {"item_id": item_id, "store_id": store_id, "horizon": 28, "price_override": price}
    r = httpx.post(f"{API}/v2/models/demand-forecaster/infer", json=body, timeout=30)
    r.raise_for_status()
    payload = r.json()
    df = pd.DataFrame(payload["forecasts"])
    df.attrs["model_version"] = payload["model_version"]
    return df.set_index("date")


with st.sidebar:
    store = st.selectbox(
        "Store", ["CA_1", "CA_2", "CA_3", "CA_4", "TX_1", "TX_2", "TX_3", "WI_1", "WI_2", "WI_3"]
    )
try:
    table = fetch_store(store)
except Exception as exc:  # API down / table empty
    st.error(f"Cannot reach the forecast API at {API}: {exc}")
    st.stop()

if table.empty:
    st.info("No batch forecasts yet - run the daily_batch_forecast DAG.")
    st.stop()

tab_view, tab_whatif = st.tabs(["Forecasts", "What-if price"])

with tab_view:
    totals = table.groupby("date")[["forecast", "lower", "upper"]].sum()
    st.subheader(f"Store {store}: total units per day (model v{table['model_version'].iloc[0]})")
    st.line_chart(totals)
    item = st.selectbox("Item", sorted(table["item_id"].unique()))
    st.line_chart(table[table["item_id"] == item].set_index("date")[["forecast", "lower", "upper"]])

with tab_whatif:
    item_w = st.selectbox("Item for what-if", sorted(table["item_id"].unique()), key="wi_item")
    base = infer(item_w, store, None)
    price = st.number_input("New price", min_value=0.01, value=5.0, step=0.25)
    changed = infer(item_w, store, price)
    cmp_df = pd.DataFrame({"baseline": base["forecast"], f"price={price:.2f}": changed["forecast"]})
    st.line_chart(cmp_df)
    delta = changed["forecast"].sum() - base["forecast"].sum()
    st.metric("28-day units", f"{changed['forecast'].sum():.1f}", f"{delta:+.1f} vs baseline")
