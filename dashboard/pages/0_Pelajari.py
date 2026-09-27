"""Pelajari -- materi dasar untuk pengguna yang belum familiar dengan saham atau LSTM."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src import config
from dashboard.components.theme import apply_theme, hero

st.set_page_config(page_title="Pelajari | Stock Dashboard", page_icon="📚", layout="wide")
apply_theme()

hero(
    "Pelajari",
    "Belum familiar dengan istilah saham atau LSTM? Mulai dari sini sebelum membuka halaman "
    "Prediction atau Historical Data.",
    badge="EDUKASI",
)

tab_saham, tab_lstm, tab_akurasi, tab_istilah = st.tabs(
    [" Dasar Saham", " Time Series & LSTM", " Membaca Akurasi", " Istilah Penting"]
)

with tab_saham:
    st.subheader("Apa itu Saham?")
    st.markdown(
        """
        **Saham** adalah bukti kepemilikan sebagian kecil dari sebuah perusahaan. Kalau kamu
        membeli saham suatu perusahaan, secara teknis kamu jadi salah satu pemilik perusahaan
        tersebut, walau porsinya bisa sangat kecil.

        Harga saham naik-turun setiap hari perdagangan karena banyak orang menawar (membeli)
        dan menjual saham tersebut di bursa. Ketika lebih banyak orang ingin membeli daripada
        menjual, harga cenderung naik -- begitu juga sebaliknya.
        """
    )

    st.subheader("Apa itu Bursa Efek Indonesia (BEI/IDX)?")
    st.markdown(
        """
        **Bursa Efek Indonesia (BEI)**, atau dalam bahasa Inggris disebut **Indonesia Stock
        Exchange (IDX)**, adalah tempat resmi di Indonesia di mana saham-saham perusahaan
        publik diperjualbelikan. Setiap perusahaan yang tercatat (*listed*) di BEI punya
        **kode saham** unik, misalnya `TLKM` untuk Telkom Indonesia atau `BBCA` untuk Bank
        Central Asia. Kode inilah yang kamu pilih di halaman **Prediction** dan
        **Historical Data**.

        Perdagangan saham di BEI hanya berlangsung pada **hari perdagangan** (Senin-Jumat,
        di luar hari libur nasional) -- karena itu istilah "hari perdagangan" sering muncul
        di dashboard ini, berbeda dengan hari kalender biasa.
        """
    )

    st.subheader("Apa itu Data OHLCV?")
    st.markdown(
        f"""
        Data historis saham biasanya dicatat sebagai **OHLCV**, singkatan dari lima kolom
        (`{", ".join(config.FEATURE_COLS)}` di dashboard ini):

        - **Open** -- harga saat sesi perdagangan hari itu dibuka.
        - **High** -- harga tertinggi yang tercapai sepanjang hari itu.
        - **Low** -- harga terendah yang tercapai sepanjang hari itu.
        - **Close** -- harga saat sesi perdagangan hari itu ditutup (ini yang diprediksi
          dashboard).
        - **Volume** -- jumlah lembar saham yang berpindah tangan hari itu.

        Kelima angka ini digabung menjadi grafik **candlestick** yang bisa kamu lihat di
        halaman Historical Data: satu "batang lilin" mewakili satu hari, dengan warna hijau
        kalau harga naik (Close > Open) dan merah kalau turun (Close < Open).
        """
    )

with tab_lstm:
    st.subheader("Apa itu Time Series (Data Deret Waktu)?")
    st.markdown(
        """
        **Time series** adalah data yang tersusun berurutan berdasarkan waktu, misalnya
        harga penutupan saham setiap hari selama beberapa tahun terakhir. Ciri khasnya:
        urutannya penting -- nilai hari ini biasanya berhubungan dengan nilai beberapa hari
        sebelumnya, bukan sekadar kumpulan angka acak.

        Memprediksi harga saham berarti mencoba menebak nilai time series di masa depan
        berdasarkan pola yang terlihat pada data historisnya.
        """
    )

    st.subheader("Apa itu LSTM?")
    st.markdown(
        """
        **LSTM (Long Short-Term Memory)** adalah salah satu jenis *neural network* yang
        dirancang khusus untuk mempelajari data berurutan seperti time series. LSTM adalah
        varian dari **RNN (Recurrent Neural Network)** -- jenis model yang memproses data
        selangkah demi selangkah sambil "mengingat" apa yang sudah dilihat sebelumnya.

        Analogi sederhana: bayangkan kamu membaca cerita bersambung. Untuk memahami bab
        terbaru, kamu perlu mengingat kejadian penting dari bab-bab sebelumnya, tapi tidak
        semua detail kecil perlu diingat. LSTM bekerja mirip seperti itu -- ia punya
        mekanisme untuk memilih informasi mana dari masa lalu yang penting untuk disimpan,
        dan mana yang boleh dilupakan, sehingga cocok untuk mengenali tren harga saham yang
        berlangsung selama puluhan hari perdagangan.
        """
    )

    st.subheader("Bagaimana Model di Dashboard Ini Bekerja?")
    st.markdown(
        f"""
        Secara singkat:

        1. Dashboard mengambil **{config.WINDOW_SIZE} hari perdagangan** data historis
           terakhir suatu saham.
        2. Data itu dimasukkan ke model LSTM yang sudah dilatih sebelumnya.
        3. Model mengeluarkan satu angka: perkiraan harga **Close** untuk hari perdagangan
           berikutnya.
        4. Untuk prediksi lebih dari satu hari ke depan, dashboard mengulang proses ini
           secara **rekursif** -- hasil prediksi hari ini dipakai sebagai bagian input untuk
           memprediksi hari berikutnya, dan seterusnya sampai tanggal target tercapai.

        Karena setiap hasil prediksi dipakai lagi sebagai input, kesalahan kecil bisa
        menumpuk semakin jauh horizon prediksinya -- karena itu dashboard membatasi prediksi
        maksimum sekitar **1 bulan (~{config.MAX_PREDICTION_HORIZON_TRADING_DAYS} hari
        perdagangan)** ke depan, dan selalu menampilkan estimasi akurasinya (lihat tab
        "Membaca Akurasi"). Penjelasan teknis lebih detail (arsitektur model, sumber data,
        cakupan saham) ada di halaman **About**.
        """
    )

with tab_akurasi:
    st.subheader("Apa itu MAPE?")
    st.markdown(
        """
        **MAPE (Mean Absolute Percentage Error)** adalah cara mengukur seberapa jauh
        prediksi model meleset dari harga yang sebenarnya terjadi, dinyatakan dalam persen.

        Misalnya, MAPE sebesar **5%** berarti secara rata-rata, prediksi model meleset
        sekitar 5% dari harga aktual. Semakin **kecil** angka MAPE, semakin **akurat**
        model tersebut untuk saham dan horizon yang dipilih.
        """
    )

    st.subheader("Apa itu RMSE?")
    st.markdown(
        """
        **RMSE (Root Mean Squared Error)** mengukur selisih prediksi vs harga aktual dalam
        satuan yang sama dengan harga saham (Rupiah), bukan persen. RMSE berguna untuk
        melihat besar selisih dalam nilai uang riil, sedangkan MAPE lebih mudah dibandingkan
        antar saham dengan rentang harga yang berbeda jauh (misalnya saham Rp 100 vs saham
        Rp 10.000).
        """
    )

    st.subheader("Kenapa Angka MAPE Berbeda-beda?")
    st.markdown(
        """
        Nilai MAPE dihitung ulang untuk **setiap kombinasi saham dan horizon prediksi**,
        karena tiap saham punya pola pergerakan harga yang berbeda, dan prediksi yang lebih
        jauh ke depan (rekursif) umumnya punya error yang lebih besar dibanding prediksi
        1 hari ke depan. Selalu perhatikan MAPE yang ditampilkan di halaman Prediction untuk
        saham & tanggal target yang kamu pilih, jangan berasumsi semua saham/horizon punya
        akurasi yang sama.
        """
    )

    st.subheader("Label Kualitatif Akurasi")
    st.markdown(
        """
        Supaya lebih mudah dibaca sekilas, dashboard memberi label warna berdasarkan MAPE:

        - 🟢 **Akurasi Tinggi** -- MAPE di bawah 5%.
        - 🟡 **Akurasi Sedang** -- MAPE antara 5% - 12%.
        - 🔴 **Akurasi Rendah** -- MAPE di atas 12%.

        Label ini hanya panduan cepat -- tetap perhatikan angka MAPE persisnya, terutama
        untuk keputusan yang penting bagimu.
        """
    )

with tab_istilah:
    st.subheader("Kamus Istilah Singkat")
    st.markdown(
        """
        | Istilah | Penjelasan Singkat |
        |---|---|
        | **Ticker / Kode Saham** | Kode unik singkat untuk sebuah saham di bursa, mis. `TLKM`, `BBCA`. |
        | **Bullish** | Kondisi saat harga cenderung naik / sentimen pasar positif. |
        | **Bearish** | Kondisi saat harga cenderung turun / sentimen pasar negatif. |
        | **Candlestick** | Grafik yang menampilkan Open, High, Low, Close dalam satu "batang lilin" per hari. |
        | **Volume** | Jumlah lembar saham yang diperdagangkan dalam satu periode waktu. |
        | **Horizon Prediksi** | Seberapa jauh ke depan (dalam hari perdagangan) model diminta memprediksi. |
        | **Backtest** | Menguji performa model pada data historis yang sudah diketahui hasilnya, untuk mengukur akurasi. |
        | **Scaler / Normalisasi** | Proses mengubah skala angka (mis. harga) ke rentang 0-1 agar mudah diproses model. |
        | **Rekursif (forecasting)** | Prediksi hari ke-t dipakai sebagai input untuk memprediksi hari ke-(t+1), dst. |
        """
    )
    st.caption(
        "Ingin penjelasan teknis yang lebih mendalam tentang model, sumber data, atau "
        "cakupan saham yang didukung? Lihat halaman **About**."
    )

st.divider()
st.info(
    "Sudah paham dasarnya? Buka halaman **Prediction** di sidebar untuk mulai memprediksi "
    "harga saham, atau **Historical Data** untuk menelusuri data historisnya."
)
