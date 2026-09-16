"""About -- penjelasan metode, sumber data, dan identitas pengembang."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import config
from dashboard.components.theme import apply_theme, hero

st.set_page_config(page_title="About | Stock Dashboard", page_icon="ℹ️", layout="wide")
apply_theme()

def hero(title, subtitle):
    st.markdown(
        f"""
        <div style="padding:20px;border-left:4px solid #00C853;background:#f8f9fa;border-radius:8px;">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Metode: Long Short-Term Memory (LSTM)")
st.markdown(
    """
    **LSTM (Long Short-Term Memory)** adalah salah satu arsitektur *Recurrent
    Neural Network* (RNN) yang dirancang khusus untuk mempelajari pola pada
    data sekuensial / *time series*, seperti pergerakan harga saham dari
    waktu ke waktu.

    Berbeda dengan RNN biasa yang cenderung "lupa" informasi dari langkah
    waktu yang jauh (*vanishing gradient*), LSTM memiliki mekanisme *gate*
    (input gate, forget gate, output gate) yang memungkinkannya menyimpan
    atau membuang informasi jangka panjang secara selektif. Karakteristik
    ini membuat LSTM cocok untuk memodelkan tren dan pola historis harga
    saham yang berlangsung selama puluhan hari perdagangan.

    **Alur kerja model pada proyek ini:**
    """
)

st.markdown(
    f"""
    1. Data historis `{", ".join(config.FEATURE_COLS)}` diambil dari Yahoo Finance.
    2. Data dinormalisasi ke rentang 0-1 menggunakan `MinMaxScaler`.
    3. Data dibentuk menjadi sequence sliding-window sepanjang
       **{config.WINDOW_SIZE} hari perdagangan** sebagai input model.
    4. Model LSTM universal (2 layer LSTM, {config.HIDDEN_SIZE} hidden units,
       dilatih atas {len(config.PRETRAINED_SCALER_CODES)} saham sekaligus)
       memprediksi harga **Close** hari berikutnya. Model ini lalu dipakai
       untuk seluruh {len(config.TICKERS)} saham IDX -- lihat bagian
       "Cakupan Saham & Kalibrasi Scaler" di bawah.
    5. Hasil prediksi (masih dalam skala 0-1) dikembalikan ke skala harga
       asli (invers dari `MinMaxScaler`).
    """
)

st.divider()

st.subheader("Prediksi Multi-Horizon (Rekursif) & MAPE")
st.markdown(
    f"""
    Model hanya dilatih untuk memprediksi **1 hari perdagangan ke depan**.
    Untuk pilihan horizon lain di halaman **Prediction**
    ({", ".join(config.HORIZON_OPTIONS.keys())}), dashboard menjalankan
    model secara **rekursif**: prediksi Close hari ke-t dipakai sebagai
    bagian dari input sequence untuk memprediksi hari ke-(t+1), dan
    seterusnya. Karena Open/High/Low/Volume hari ke-t yang sebenarnya
    belum diketahui, dashboard memakai asumsi sederhana: Open = High = Low
    = Close = harga prediksi, dan Volume mengikuti nilai terakhir yang
    diketahui.

    Konsekuensinya, **error cenderung terakumulasi semakin jauh
    horizonnya**. Karena itu, setiap horizon dievaluasi ulang lewat
    *backtest* pada data historis (test set 15% terakhir, tidak pernah
    dipakai untuk training) untuk menghasilkan **MAPE (Mean Absolute
    Percentage Error)** -- ditampilkan di halaman Prediction supaya kamu
    tahu seberapa reliabel prediksi untuk horizon yang dipilih.
    """
)

st.divider()

st.subheader("Sumber Data")
st.markdown(
    """
    Seluruh data harga saham diambil dari **Yahoo Finance** melalui pustaka
    Python [`yfinance`](https://pypi.org/project/yfinance/). Data yang
    digunakan mencakup kolom `Date`, `Open`, `High`, `Low`, `Close`, dan
    `Volume`.

    - Data pelatihan model: 5 tahun terakhir per saham.
    - Data pada halaman **Prediction** & **Historical Data**: diambil
      *real-time* setiap kali halaman dibuka -- dashboard tidak menyimpan
      salinan harian ke database.
    """
)

st.divider()

st.subheader("Cakupan Saham & Kalibrasi Scaler Otomatis")
st.markdown(
    f"""
    Model LSTM di atas dilatih atas data historis **{len(config.PRETRAINED_SCALER_CODES)}
    saham** -- untuk saham-saham ini, scaler `MinMaxScaler` sudah di-fit
    sebelumnya dan dibundel bersama model (`scalers/scalers.pkl`), jadi
    langsung bisa dipakai.

    Dashboard ini kemudian **diperluas ke seluruh {len(config.TICKERS)} saham**
    yang tercatat di Bursa Efek Indonesia. Untuk saham di luar
    {len(config.PRETRAINED_SCALER_CODES)} saham training tadi, scaler
    dikalibrasi **secara otomatis** saat pertama kali saham tersebut
    diminta:

    1. Dashboard mengunduh data historis {config.TRAIN_PERIOD} saham
       tersebut dari Yahoo Finance.
    2. `MinMaxScaler` di-fit pada 70% data pertama (porsi train) --
       metodologi yang sama seperti saat melatih model, supaya tidak ada
       kebocoran data dari test set.
    3. Scaler hasil kalibrasi disimpan ke `scalers/dynamic/<KODE>.pkl`
       supaya permintaan berikutnya untuk saham yang sama tidak perlu
       mengulang proses ini.

    Karena arsitektur LSTM-nya sama dan datanya dinormalisasi dengan cara
    yang identik, model bisa dipakai untuk saham di luar data trainingnya
    -- tapi **akurasinya belum tentu sama** dengan saham yang benar-benar
    ada di data training. Selalu perhatikan **MAPE hasil backtest** di
    halaman Prediction untuk saham yang kamu pilih (backtest berjalan
    untuk semua saham, pretrained maupun auto-kalibrasi, jadi angka MAPE
    yang ditampilkan selalu spesifik untuk saham tersebut).
    """
)

st.divider()

st.subheader("Daftar Saham")
st.markdown(
    f"Dashboard ini mendukung **{len(config.TICKERS)} saham** di Bursa Efek Indonesia,"
    f"{len(config.PRETRAINED_SCALER_CODES)} di antaranya termasuk data training model "
)
_search = st.text_input(" Cari kode saham", placeholder="mis. ADRO, BBCA, TLKM...")
_filtered = [
    code for code in config.TICKERS
    if _search.strip().upper() in code
] if _search else list(config.TICKERS.keys())
st.dataframe(
    {
        "Kode Saham": _filtered,
        "Nama Perusahaan": [config.STOCK_NAMES.get(c, "") for c in _filtered],
        "Ticker Yahoo Finance": [config.TICKERS[c] for c in _filtered],
        "Status Scaler": [
            "✅ Pretrained" if c in config.PRETRAINED_SCALER_CODES else "🆕 Auto-kalibrasi"
            for c in _filtered
        ],
    },
    width="stretch",
    height=300,
    hide_index=True,
)


st.divider()

st.subheader("Batasan & Disclaimer")
st.warning(
    """
    - Prediksi model bersifat **estimasi statistik** berdasarkan pola data
      historis, dan **tidak memperhitungkan** berita, sentimen pasar,
      kebijakan perusahaan, kondisi makroekonomi, maupun faktor fundamental
      lainnya.
    - Performa model di masa lalu (pada data historis / test set) **tidak
      menjamin** akurasi prediksi di masa depan.
    - Dashboard ini dibuat untuk **tujuan edukasi/riset** dan bukan
      merupakan rekomendasi maupun nasihat investasi. Selalu lakukan riset
      mandiri dan konsultasikan keputusan finansial dengan ahli yang
      berkompeten.
    """
)

st.divider()

st.subheader(" Identitas Pengembang")

st.markdown("""
Dashboard ini dikembangkan sebagai bagian dari proyek Praktik Kerja Lapangan (PKL)
mengenai **Prediksi Harga Saham menggunakan Long Short-Term Memory (LSTM)** berbasis
**Streamlit**.

### Pengembang
- **Nama:** Dwi Fajar Novianto
- **Program Studi:** Matematika
- **Institusi:** Universitas Negeri Yogyakarta

### Kontak
- 🌐 **GitHub:** https://github.com/Bigg-2025
- 💼 **LinkedIn:** https://www.linkedin.com/in/dwi-fajar-novianto
""")
