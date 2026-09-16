"""Prediction -- pilih saham & horizon, ambil data terbaru, tampilkan prediksi + MAPE."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import config, predict
from src.stock_names import get_stock_name
from dashboard.components.charts import forecast_path_chart, line_close_chart, mape_by_horizon_chart
from dashboard.components.formatting import format_rupiah, format_date_id
from dashboard.components.theme import apply_theme, hero, accuracy_pill, quote_card

st.set_page_config(page_title="Prediction | Stock Dashboard", layout="wide")
apply_theme()

hero(
    "Prediction",
    "Pilih saham dan horizon prediksi, lalu jalankan model LSTM dengan data terbaru dari Yahoo Finance. "
    "Mendukung seluruh saham yang tercatat di Bursa Efek Indonesia.",
    badge="LSTM INFERENCE",
)

# ---------------------------------------------------------------------------
# Pemilihan saham & horizon
# ---------------------------------------------------------------------------
# Nama perusahaan diambil dari data/idx_stock_list.csv yang dibundel offline
# (bukan Yahoo Finance) -- jadi aman ditampilkan untuk semua ~950 opsi
# dropdown sekaligus tanpa network call.
stock_options = list(config.TICKERS.keys())
stock_labels = {
    code: f"{code} - {config.STOCK_NAMES.get(code, config.TICKERS[code])}"
    for code in stock_options
}
horizon_labels = list(config.HORIZON_OPTIONS.keys())

with st.container(border=True):
    col_stock, col_horizon = st.columns([1.4, 2])
    with col_stock:
        stock_code = st.selectbox(
            f" Pilih Saham ({len(stock_options)} tersedia, ketik untuk mencari)",
            options=stock_options,
            format_func=lambda c: stock_labels[c],
        )
        if predict.is_pretrained_stock(stock_code):
            st.caption("Termasuk saham yang dipakai training model.")
        else:
            st.caption(
                "Di luar data training model."
            )
    with col_horizon:
        st.markdown("Horizon Prediksi")
        horizon_label = st.segmented_control(
            "Horizon Prediksi",
            options=horizon_labels,
            default=config.DEFAULT_HORIZON_LABEL,
            label_visibility="collapsed",
        ) or config.DEFAULT_HORIZON_LABEL

    run_prediction = st.button(" Jalankan Prediksi", type="primary", width="stretch")

horizon_days = config.HORIZON_OPTIONS[horizon_label]


@st.cache_data(ttl=300, show_spinner=False)
def _cached_fetch(code: str):
    return predict.fetch_latest_data(code)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_forecast(code: str, h: int):
    history = _cached_fetch(code)
    return predict.forecast(code, h, history=history)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_all_horizons_mape(code: str):
    return predict.backtest_all_horizons(code)


@st.cache_data(ttl=86400, show_spinner=False)
def _cached_stock_name(code: str):
    return get_stock_name(code)


if run_prediction:
    st.session_state["prediction_stock"] = stock_code
    st.session_state["prediction_horizon"] = horizon_label

active_stock = st.session_state.get("prediction_stock")
active_horizon_label = st.session_state.get("prediction_horizon", config.DEFAULT_HORIZON_LABEL)
active_horizon_days = config.HORIZON_OPTIONS[active_horizon_label]

if not active_stock:
    st.info(" Pilih saham & horizon, lalu klik **Jalankan Prediksi** untuk memulai.")
else:
    ticker = config.TICKERS[active_stock]
    with st.spinner(f"Mengambil data terbaru {ticker} dari Yahoo Finance..."):
        try:
            history = _cached_fetch(active_stock)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal mengambil data dari Yahoo Finance untuk **{ticker}**.\n\nDetail: {exc}")
            st.stop()

    _spinner_msg = f"Menjalankan model LSTM untuk horizon {active_horizon_label}..."
    if not predict.is_pretrained_stock(active_stock):
        _spinner_msg += " (kalibrasi scaler otomatis untuk saham ini, mohon tunggu...)"
    with st.spinner(_spinner_msg):
        try:
            result = _cached_forecast(active_stock, active_horizon_days)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gagal menjalankan prediksi: {exc}")
            st.stop()

    st.success(
        f"Prediksi **{active_stock}** ({ticker}) untuk horizon **{active_horizon_label}** berhasil dihitung."
    )

    # -----------------------------------------------------------------
    # Kartu harga (quote card) -- ala halaman detail saham
    # -----------------------------------------------------------------
    quote_card(
        ticker=active_stock,
        company_name=_cached_stock_name(active_stock),
        last_label=f"Harga Terakhir ({format_date_id(result.last_date)})",
        last_value_str=format_rupiah(result.last_close),
        predicted_label=f"Prediksi {active_horizon_label} ({format_date_id(result.predicted_date)})",
        predicted_value_str=format_rupiah(result.predicted_close),
        change_pct=result.change_pct,
        meta_caption=f"{ticker} &middot; Bursa Efek Indonesia",
    )

    # -----------------------------------------------------------------
    # MAPE / akurasi model untuk horizon terpilih
    # -----------------------------------------------------------------
    st.divider()
    st.subheader(" Akurasi Model untuk Horizon Ini")

    if result.mape is not None:
        mc1, mc2, mc3 = st.columns([1, 1, 2])
        mc1.metric("MAPE (Test Set)", f"{result.mape:.2f}%")
        mc2.metric("RMSE (Test Set)", format_rupiah(result.rmse))
        with mc3:
            st.markdown(
                f"{accuracy_pill(result.mape)} dievaluasi pada **{result.n_backtest_samples} sample** "
                f"data historis (test set) yang tidak dipakai saat training.",
                unsafe_allow_html=True,
            )
        st.caption(
            "MAPE (Mean Absolute Percentage Error) mengukur rata-rata persentase selisih "
            "antara harga prediksi dan harga aktual. Semakin kecil, semakin akurat. "
            "Untuk horizon > 1 hari, model memprediksi secara **rekursif** (hasil hari "
            "sebelumnya dipakai sebagai input hari berikutnya), sehingga error cenderung "
            "membesar seiring bertambahnya horizon -- lihat grafik perbandingan di bawah."
        )

        with st.expander(" Bandingkan MAPE di semua horizon"):
            with st.spinner("Menghitung backtest untuk semua horizon..."):
                all_mape = _cached_all_horizons_mape(active_stock)
            labels = [lbl for lbl in horizon_labels if all_mape.get(lbl)]
            values = [all_mape[lbl]["mape"] for lbl in labels]
            st.plotly_chart(
                mape_by_horizon_chart(labels, values, active_label=active_horizon_label),
                width="stretch",
            )
    else:
        st.info("MAPE tidak tersedia untuk kombinasi saham/horizon ini (data historis lokal tidak cukup).")

    st.divider()

    # -----------------------------------------------------------------
    # Grafik
    # -----------------------------------------------------------------
    tab1, tab2 = st.tabs(["📈 Historis + Prediksi", "📉 Historis (garis)"])
    with tab1:
        st.plotly_chart(
            forecast_path_chart(result.history, result.forecast_dates, result.forecast_closes),
            width="stretch",
        )
        if active_horizon_days > 1:
            st.caption(
                f"Garis putus-putus oranye menunjukkan lintasan prediksi harian dari sekarang "
                f"hingga {active_horizon_days} hari perdagangan ke depan; bintang menandai target "
                f"horizon ({format_date_id(result.predicted_date)})."
            )
            with st.expander(" Lihat rincian prediksi harian"):
                path_df = result.as_path_df().copy()
                path_df["Tanggal"] = path_df["Date"].apply(format_date_id)
                path_df["Harga Prediksi"] = path_df["Close"].apply(format_rupiah)
                st.dataframe(
                    path_df[["Tanggal", "Harga Prediksi"]],
                    width="stretch",
                    hide_index=True,
                )
    with tab2:
        st.plotly_chart(
            line_close_chart(result.history, title=f"Harga Historis {active_stock}"),
            width="stretch",
        )

    st.caption(
        "⚠️ Ini adalah estimasi statistik dari model LSTM, bukan rekomendasi atau nasihat "
        "investasi. Pergerakan pasar riil dipengaruhi banyak faktor yang tidak sepenuhnya "
        "tertangkap oleh model, dan prediksi horizon panjang bersifat rekursif sehingga "
        "lebih tidak pasti dibanding horizon pendek."
    )
