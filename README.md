# Stock Prediction Dashboard

Dashboard prediksi harga saham menggunakan **Long Short-Term Memory (LSTM)**
berbasis **Streamlit**, dengan tampilan modern (tema custom, hero banner,
kartu metrik) dan prediksi **multi-horizon** (1 hari s/d 3 bulan) lengkap
dengan estimasi akurasi **MAPE** per horizon. Data historis diambil otomatis
dari Yahoo Finance.

##  Fitur Utama

- **Prediksi multi-horizon** -- 1 Hari, 3 Hari, 1 Minggu, 2 Minggu, hingga
  1 Bulan. Untuk horizon > 1 hari, model (yang hanya dilatih 1-hari-ke-depan)
  dipakai secara **rekursif**: prediksi hari ke-t jadi bagian input untuk
  memprediksi hari ke-(t+1).
- **MAPE per horizon** -- setiap horizon dievaluasi lewat *backtest* pada
  test set historis (data yang tidak dipakai training), jadi kamu bisa
  lihat langsung seberapa reliabel prediksi untuk horizon yang dipilih
  (semakin jauh horizon, MAPE biasanya semakin besar -- ini ditampilkan
  apa adanya, bukan disembunyikan).
- **Tampilan modern ala aplikasi trading saham** -- terinspirasi Stockbit:
  header bersih dengan aksen hijau, kartu "quote" harga besar + pill
  hijau/merah untuk naik/turun (bukan `st.metric` yang bisa terpotong),
  layout memakai lebar layar penuh di desktop (`dashboard/assets/style.css`
  + `dashboard/components/theme.py`).

## Catatan Penyesuaian dari Rencana Awal

Dokumen rencana proyek menyebutkan saham **BREN, BYAN, TLKM** dan model
berbasis **TensorFlow/Keras (.keras)**. File yang diunggah (`Dashboard.zip`)
ternyata berisi model dan scaler yang sudah dilatih untuk saham berbeda,
menggunakan framework berbeda:

| Rencana awal | Yang tersedia di `Dashboard.zip` |
|---|---|
| BREN, BYAN, TLKM | **TLKM, BYAN, ASII** |
| Model `.keras` (TensorFlow) | Model `.pt` (**PyTorch**, `state_dict`) |

Dashboard ini dibangun mengikuti **artefak yang benar-benar tersedia**
(TLKM, BYAN, ASII / PyTorch), karena itulah model yang sudah dilatih dan bisa
langsung dipakai untuk inference. Menambahkan saham **BREN** sangat mudah
kalau modelnya sudah dilatih -- lihat bagian [Menambah Saham Baru](#menambah-saham-baru).

Satu file model punya salah ketik dari sumber aslinya
(`best_lstm_ASSI.pt` -> seharusnya "ASII"); di proyek ini sudah
diganti namanya menjadi `best_lstm_ASII.pt` agar konsisten dengan
`scaler_ASII.pkl` dan kode saham `ASII`.

## Struktur Proyek

```
stock_prediction_dashboard/
├── .streamlit/
│   └── config.toml            # tema warna native Streamlit
│
├── data/
│   ├── idx_stock_list.csv     # master list ~951 kode + nama saham IDX
│   ├── raw/                   # cache CSV historis (hasil src/download_data.py / backtest)
│   └── processed/             # cache nama saham fallback (stock_names_cache.json)
│
├── models/                    # model LSTM (.pt legacy per-saham, .pth universal)
├── scalers/                   # MinMaxScaler: scalers.pkl (pretrained, ~300 saham)
│   └── dynamic/                # cache scaler auto-kalibrasi (saham di luar training)
├── notebooks/                  # notebook eksperimen asli (referensi/dokumentasi)
│
├── src/
│   ├── config.py               # path, daftar saham (TICKERS, PRETRAINED_SCALER_CODES), hyperparameter
│   ├── model.py                 # UniversalLSTM + legacy LSTMRegressor
│   ├── stock_names.py            # nama perusahaan (offline CSV + fallback Yahoo Finance)
│   ├── download_data.py         # unduh data training dari Yahoo Finance (legacy)
│   ├── preprocess.py            # scaling, sliding-window sequence, inverse-transform
│   ├── train.py                  # training pipeline legacy (offline, tidak dipanggil dashboard)
│   └── predict.py                # forecast() rekursif multi-horizon + backtest_mape() + dynamic scaler
│
├── dashboard/
│   ├── app.py                    # Home
│   ├── pages/
│   │   ├── 1_Prediction.py        # pilih horizon, prediksi + MAPE
│   │   ├── 2_Historical_Data.py
│   │   └── 3_About.py
│   ├── components/
│   │   ├── charts.py               # line, candlestick, volume, forecast path, MAPE bar
│   │   ├── formatting.py           # format Rupiah, persen, tanggal Indonesia
│   │   └── theme.py                 # apply_theme(), hero(), accuracy_pill()
│   └── assets/
│       └── style.css                # CSS tema modern (gradient, kartu, badge)
│
├── requirements.txt
└── README.md
```

## Instalasi

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard **tidak melakukan training** saat dijalankan -- ia hanya memuat
model & scaler yang sudah ada di `models/` dan `scalers/`, lalu:

```
Yahoo Finance -> download data terbaru -> preprocessing -> load scaler
              -> load model -> prediksi -> visualisasi
```

Setiap kali halaman **Prediction** atau **Historical Data** dibuka, data
terbaru diambil langsung dari Yahoo Finance (di-cache 5 menit lewat
`st.cache_data` agar tidak membebani API saat navigasi berulang).

### Halaman

- **Home** -- deskripsi aplikasi, metode LSTM, daftar saham, info proyek.
- **Prediction** -- pilih saham & horizon (1 Hari s/d 1 Bulan), ambil data
  terbaru, tampilkan harga terakhir, harga prediksi, persentase perubahan,
  tanggal target, **MAPE & RMSE (test set)** untuk horizon terpilih, grafik
  perbandingan MAPE di semua horizon, serta grafik historis + lintasan
  prediksi.
- **Historical Data** -- data historis dalam line chart, candlestick chart,
  volume perdagangan, dan tabel data (bisa diunduh sebagai CSV).
- **About** -- penjelasan metode LSTM, sumber data, dan identitas
  pengembang (silakan sesuaikan bagian identitas di
  `dashboard/pages/3_About.py`).


## Model Universal (Seluruh Saham IDX)

Dashboard memakai **satu model LSTM universal**
(`models/universal_lstm_best.pth`) yang dilatih sekaligus atas data
historis ~300 saham, alih-alih satu model terpisah per saham. Model ini
kemudian **diperluas ke seluruh saham yang tercatat di Bursa Efek
Indonesia** (~951 saham, lihat `data/idx_stock_list.csv`):

- **~300 saham** (data training) -- scaler `MinMaxScaler` sudah di-fit
  sebelumnya dan dibundel dalam `scalers/scalers.pkl`
  (`dict {kode_saham: MinMaxScaler}`), langsung dipakai.
- **Sisanya (~650 saham)** -- di luar data training model, scaler
  dikalibrasi **otomatis** saat pertama kali saham tersebut diminta:
  dashboard mengunduh data historis `config.TRAIN_PERIOD` (default 5
  tahun) dari Yahoo Finance, lalu fit `MinMaxScaler` pada 70% data
  pertama (porsi train) -- metodologi sama seperti training, tidak ada
  kebocoran data test set. Hasilnya di-cache ke
  `scalers/dynamic/<KODE>.pkl` supaya permintaan berikutnya untuk saham
  yang sama instan. Lihat `src/predict.py::_get_or_fit_dynamic_scaler()`.

Karena arsitekturnya sama dan normalisasi datanya identik, model bisa
menghasilkan prediksi untuk saham di luar data trainingnya -- tapi
**akurasinya tidak dijamin sama** dengan saham yang memang ada di data
training. Backtest MAPE tetap dihitung otomatis untuk saham manapun yang
dipilih (lihat `predict.backtest_mape`), jadi selalu cek angka MAPE yang
ditampilkan di halaman Prediction sebelum mempercayai hasilnya.

- Arsitektur: 2 layer LSTM terpisah (64 hidden units masing-masing) diikuti
  `Linear(64,16) -> ReLU -> Dropout -> Linear(16,1)`, lihat
  `src/model.py::UniversalLSTM`.
- Window size: 60 hari perdagangan (asumsi, sama seperti model lama --
  **sesuaikan `WINDOW_SIZE` di `src/config.py` jika training model
  universal memakai window berbeda**, karena `.pth` tidak menyimpan
  informasi ini).
- Fitur input: `Open, High, Low, Close, Volume`.
- Target: `Close` hari perdagangan berikutnya.
- Daftar saham (`src/config.TICKERS`) dibangun dari
  `data/idx_stock_list.csv` (master list kode + nama ~951 saham IDX,
  diolah dari dataset
  [wildangunawan/Dataset-Saham-IDX](https://github.com/wildangunawan/Dataset-Saham-IDX),
  per Januari 2025, lisensi CC BY-NC 4.0 -- non-komersial/edukasi saja)
  digabung dengan kunci-kunci di `scalers.pkl`
  (`config.PRETRAINED_SCALER_CODES`). Daftar ini mungkin belum mencakup
  IPO terbaru (setelah Januari 2025) atau saham yang sudah delisting --
  perbarui `data/idx_stock_list.csv` sesuai kebutuhan.
- Nama perusahaan diambil dari `data/idx_stock_list.csv` (offline,
  instan); untuk kode yang tidak ada di file itu, fallback ke Yahoo
  Finance dengan cache di `data/processed/stock_names_cache.json` (lihat
  `src/stock_names.py`).

### Mode Legacy (3 Saham: TLKM, BYAN, ASII)

Pipeline lama (satu model per saham) masih ada di `src/train.py` dan
`src/download_data.py` untuk keperluan retraining, memakai
`config.LEGACY_TICKERS` dan `config.legacy_model_path()` /
`config.legacy_scaler_path()`. Dashboard (`dashboard/`) **tidak lagi**
memakainya -- semua halaman sekarang memanggil model universal.

## Cara Kerja Prediksi Multi-Horizon (Penting)

Model LSTM (`src/model.py`) hanya dilatih untuk memprediksi **Close 1 hari
ke depan** dari 60 hari OHLCV historis. Untuk horizon lebih panjang, dipakai
strategi **rekursif** (`src/predict.py -> recursive_forecast_batch`):

1. Prediksi Close hari ke-1 dari 60 hari data historis nyata.
2. Bentuk "hari sintetis" untuk hari ke-1: `Open = High = Low = Close =
   harga prediksi`, `Volume = volume terakhir yang diketahui`.
3. Geser window (buang hari terlama, tambahkan hari sintetis), lalu
   prediksi hari ke-2. Ulangi sampai horizon tercapai.

Ini adalah pendekatan standar untuk model 1-step-ahead yang dipakai
multi-step, tapi **error bisa terakumulasi** semakin jauh horizonnya --
karena itu setiap horizon punya MAPE sendiri dari backtest, bukan angka
tunggal yang diklaim berlaku untuk semua horizon. Selalu perhatikan nilai
MAPE yang ditampilkan sebelum menginterpretasikan hasil prediksi horizon
panjang.

Untuk model universal, backtest MAPE mengunduh data historis
(`config.TRAIN_PERIOD`, default 5 tahun) dari Yahoo Finance per saham dan
meng-cache-nya ke `data/raw/<KODE>.csv` -- butuh koneksi internet pada
pemanggilan pertama untuk tiap saham.

## Disclaimer


Prediksi yang ditampilkan bersifat estimatif berdasarkan pola data historis
dan **bukan merupakan rekomendasi atau nasihat investasi**. Selalu lakukan
riset mandiri sebelum mengambil keputusan finansial.
