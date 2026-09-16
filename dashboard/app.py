"""
Home -- Stock Prediction Dashboard

Jalankan dari root folder proyek:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import config
from dashboard.components.theme import apply_theme, hero

st.set_page_config(
    page_title="Stock Prediction Dashboard",
    layout="wide",
)
apply_theme()

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
hero(
    " Stock Prediction Dashboard",
    "Prediksi harga penutupan saham untuk beberapa hari ke depan menggunakan "
    "Long Short-Term Memory (LSTM), mendukung seluruh saham di Bursa Efek Indonesia, "
    "lengkap dengan estimasi akurasi (MAPE) tiap horizon.",
    badge="LSTM • STREAMLIT • YAHOO FINANCE",
)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Tentang Aplikasi")
    st.markdown(
        f"""
        Dashboard ini memprediksi **harga penutupan (Close Price)** hari
        perdagangan berikutnya untuk **seluruh {len(config.TICKERS)} saham**
        yang tercatat di Bursa Efek Indonesia, menggunakan model **LSTM
        universal** yang dilatih sekaligus atas data historis
        {len(config.PRETRAINED_SCALER_CODES)} saham.

        Data historis diambil **otomatis dan real-time** dari Yahoo Finance
        setiap kali halaman prediksi dibuka -- dashboard ini tidak melatih
        ulang model, hanya melakukan *inference* dengan model yang sudah
        dilatih sebelumnya.
        """
    )

    st.subheader("Metode: LSTM")
    st.markdown(
        """
        **LSTM (Long Short-Term Memory)** adalah varian dari Recurrent
        Neural Network (RNN) yang dirancang untuk mempelajari pola jangka
        panjang pada data sekuensial/time series, seperti pergerakan harga
        saham dari hari ke hari.

        Dashboard ini memakai **satu model LSTM universal** yang dilatih
        sekaligus atas data historis {n_pretrained} saham di Bursa Efek
        Indonesia -- bukan satu model per saham. Model ini kemudian
        **diperluas ke seluruh {n_all} saham** yang tercatat di BEI: untuk
        saham di luar {n_pretrained} saham training, scaler (MinMaxScaler)
        dikalibrasi otomatis dari data historis saham tersebut saat pertama
        kali diminta, memakai metodologi yang sama seperti saat training.
        """.format(n_pretrained=len(config.PRETRAINED_SCALER_CODES), n_all=len(config.TICKERS))
    )

    st.subheader("Prediksi Multi-Horizon + Estimasi Akurasi")
    st.markdown(
        f"""
        Model dilatih untuk memprediksi **1 hari perdagangan ke depan**.
        Untuk melihat prediksi yang lebih jauh ({", ".join(config.HORIZON_OPTIONS.keys())}),
        dashboard menjalankan model secara **rekursif** (prediksi hari ini
        dipakai sebagai bagian input untuk memprediksi hari berikutnya).
        """
    )

with col2:
    st.subheader("Daftar Saham")
    with st.container(border=True):
        cA, cB = st.columns(2)
        cA.metric("Total saham", f"{len(config.TICKERS)}")
        cB.metric("Scaler pretrained", f"{len(config.PRETRAINED_SCALER_CODES)}")
        st.caption(
            f"Model universal mendukung seluruh {len(config.TICKERS)} saham IDX. "
            f"{len(config.PRETRAINED_SCALER_CODES)} di antaranya termasuk data training "
            "(scaler sudah teruji); sisanya dikalibrasi otomatis saat pertama diminta. "
            "Cari kode saham di halaman **Prediction** atau **Historical Data**."
        )
        with st.expander("Lihat semua kode saham"):
            codes = list(config.TICKERS.keys())
            n_cols = 4
            cols = st.columns(n_cols)
            for i, code in enumerate(codes):
                mark = "✅" if code in config.PRETRAINED_SCALER_CODES else "🆕"
                cols[i % n_cols].markdown(f"{mark} `{code}`")

    st.subheader("Ringkasan Model")
    with st.container(border=True):
        st.markdown(
            f"""
            - Arsitektur: LSTM universal (2 layer LSTM, {config.HIDDEN_SIZE} hidden units) + FC {config.FC_HIDDEN_SIZE}
            - Window size: {config.WINDOW_SIZE} hari perdagangan
            - Fitur input: {", ".join(config.FEATURE_COLS)}
            - Target: {config.TARGET_COL} hari berikutnya
            - Scaler: MinMaxScaler (0-1), satu scaler per saham
              ({len(config.PRETRAINED_SCALER_CODES)} pretrained, sisanya auto-kalibrasi)
            - Jumlah saham didukung: {len(config.TICKERS)} (seluruh saham IDX)
            - Horizon tersedia: {", ".join(config.HORIZON_OPTIONS.keys())}
            """
        )

st.divider()
st.markdown(
    " Buka halaman **Prediction** di sidebar kiri untuk melihat prediksi "
    "harga hari berikutnya, atau **Historical Data** untuk menelusuri data "
    "historis masing-masing saham."
)


