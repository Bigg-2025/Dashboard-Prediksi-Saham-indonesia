"""Historical Data -- data historis, line chart, candlestick, dan volume perdagangan."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import config, predict
from src.stock_names import get_stock_name
from dashboard.components.charts import candlestick_chart, line_close_chart, volume_chart
from dashboard.components.formatting import format_rupiah
from dashboard.components.theme import apply_theme, hero

st.set_page_config(page_title="Historical Data | Stock Dashboard", layout="wide")
apply_theme()

hero(
    " Historical Data",
    "Telusuri data historis harga saham: line chart, candlestick, volume, dan tabel data.",
    badge="YAHOO FINANCE",
)

options = list(config.TICKERS.keys())
labels = {
    code: f"{code} - {config.STOCK_NAMES.get(code, config.TICKERS[code])}"
    for code in options
}

with st.container(border=True):
    col1, col2 = st.columns([2, 2])
    with col1:
        stock_code = st.selectbox(
            f" Pilih Saham ({len(options)} tersedia, ketik untuk mencari)",
            options=options,
            format_func=lambda c: labels[c],
        )
    with col2:
        range_option = st.selectbox(
            " Rentang Waktu",
            options=["3 Bulan", "6 Bulan", "1 Tahun", "2 Tahun", "5 Tahun", "Semua"],
            index=2,
        )

RANGE_DAYS = {
    "3 Bulan": 63, "6 Bulan": 126, "1 Tahun": 252,
    "2 Tahun": 504, "5 Tahun": 1260, "Semua": None,
}


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fetch(code: str):
    return predict.fetch_latest_data(code, period="5y")


with st.spinner(f"Mengambil data historis {config.TICKERS[stock_code]} dari Yahoo Finance..."):
    try:
        df = _cached_fetch(stock_code)
    except Exception as exc:  # noqa: BLE001
        st.error(
            f"Gagal mengambil data dari Yahoo Finance untuk **{config.TICKERS[stock_code]}**.\n\n"
            f"Detail: {exc}"
        )
        st.stop()

n_days = RANGE_DAYS[range_option]
view_df = df.tail(n_days) if n_days else df


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_stock_name(code: str):
    return get_stock_name(code)


st.caption(f"**{stock_code}** &middot; {_cached_stock_name(stock_code)} &middot; `{config.TICKERS[stock_code]}`")

# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Harga Terakhir", format_rupiah(view_df["Close"].iloc[-1]))
c2.metric("Tertinggi (periode)", format_rupiah(view_df["High"].max()))
c3.metric("Terendah (periode)", format_rupiah(view_df["Low"].min()))
c4.metric("Rata-rata Volume", f"{view_df['Volume'].mean():,.0f}".replace(",", "."))

st.divider()

tab1, tab2, tab3, tab4 = st.tabs(["Line Chart", " Candlestick", " Volume", " Tabel Data"])

with tab1:
    st.plotly_chart(line_close_chart(view_df, title=f"Harga Penutupan {stock_code}"), width="stretch")

with tab2:
    st.plotly_chart(candlestick_chart(view_df, title=f"Candlestick {stock_code}"), width="stretch")

with tab3:
    st.plotly_chart(volume_chart(view_df, title=f"Volume Perdagangan {stock_code}"), width="stretch")

with tab4:
    display_df = view_df.copy()
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    st.dataframe(
        display_df.sort_values("Date", ascending=False).reset_index(drop=True),
        width="stretch",
        height=500,
    )
    st.download_button(
        "Unduh CSV",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{stock_code}_historical.csv",
        mime="text/csv",
    )
