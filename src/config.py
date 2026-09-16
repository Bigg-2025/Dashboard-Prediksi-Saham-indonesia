"""
Konfigurasi pusat untuk Stock Prediction Dashboard.

Semua path, daftar saham, dan hyperparameter model didefinisikan di sini
agar training pipeline (src/train.py) dan dashboard (dashboard/app.py)
selalu konsisten satu sama lain.
"""

from pathlib import Path

import csv
import joblib

# ---------------------------------------------------------------------------
# Path dasar proyek
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = BASE_DIR / "scalers"
DYNAMIC_SCALERS_DIR = SCALERS_DIR / "dynamic"

for _dir in (DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, SCALERS_DIR, DYNAMIC_SCALERS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model universal
# ---------------------------------------------------------------------------
# Sejak versi ini, dashboard memakai SATU model LSTM universal (dilatih atas
# ~300 saham sekaligus) alih-alih satu model per saham. File scaler tetap
# per-saham karena rentang harga tiap saham berbeda jauh, tapi semuanya
# dibundel dalam satu file scalers.pkl berisi dict {kode_saham: MinMaxScaler}.
UNIVERSAL_MODEL_PATH = MODELS_DIR / "universal_lstm_best.pth"
SCALERS_PATH = SCALERS_DIR / "scalers.pkl"

# ---------------------------------------------------------------------------
# Daftar saham
# ---------------------------------------------------------------------------
# Dashboard mendukung SELURUH saham yang tercatat di Bursa Efek Indonesia,
# tidak dibatasi hanya ~300 saham yang dipakai saat training model universal.
# TICKERS dibangun dari data/idx_stock_list.csv, master list ~950 kode saham
# IDX beserta nama perusahaannya (sumber: data historis BEI, lihat README
# untuk atribusi). Untuk saham yang TIDAK termasuk dalam ~300 saham yang
# dipakai melatih model (lihat PRETRAINED_SCALER_CODES di bawah), scaler
# dikalibrasi otomatis saat pertama kali diminta -- lihat
# src/predict.py::_get_or_fit_dynamic_scaler().
IDX_STOCK_LIST_PATH = DATA_RAW_DIR.parent / "idx_stock_list.csv"


def _load_idx_stock_list() -> dict:
    """Baca data/idx_stock_list.csv -> {kode: nama_perusahaan}."""
    if not IDX_STOCK_LIST_PATH.exists():
        return {}
    names = {}
    with open(IDX_STOCK_LIST_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            names[row["code"].strip().upper()] = row["name"].strip()
    return names


def _load_pretrained_scaler_codes() -> set:
    """Kode saham yang scaler-nya sudah di-fit sebelumnya (dibundel di
    scalers.pkl) -- ini adalah ~300 saham yang datanya dipakai melatih
    model universal, jadi paling bisa diandalkan."""
    if not SCALERS_PATH.exists():
        return set()
    scalers_dict = joblib.load(SCALERS_PATH)
    return set(scalers_dict.keys())


STOCK_NAMES = _load_idx_stock_list()
PRETRAINED_SCALER_CODES = _load_pretrained_scaler_codes()

# Kode saham yang didukung dashboard = union antara master list IDX dan
# kode yang punya scaler pretrained (berjaga-jaga kalau ada mismatch).
_ALL_CODES = sorted(set(STOCK_NAMES.keys()) | PRETRAINED_SCALER_CODES)

TICKERS = {code: f"{code}.JK" for code in _ALL_CODES}

# ---------------------------------------------------------------------------
# Fitur & preprocessing (harus SAMA PERSIS dengan proses training)
# ---------------------------------------------------------------------------
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]
TARGET_COL = "Close"
TARGET_IDX = FEATURE_COLS.index(TARGET_COL)

# PENTING: WINDOW_SIZE harus sama persis dengan window size yang dipakai
# saat melatih universal_lstm_best.pth. Nilai 60 diwariskan dari pipeline
# model per-saham sebelumnya -- jika model universal dilatih dengan window
# size berbeda, ganti nilai ini (arsitektur LSTM tidak menyimpan window
# size di dalam state_dict, jadi tidak bisa dideteksi otomatis dari file
# model).
WINDOW_SIZE = 60
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.85          # kumulatif: 70% train, 15% val, 15% test

# Periode data historis yang diambil dari Yahoo Finance
TRAIN_PERIOD = "5y"
# Untuk inference kita hanya butuh sedikit lebih dari WINDOW_SIZE hari kerja,
# tapi diminta lebih longgar supaya tetap cukup walau ada hari libur bursa.
INFERENCE_LOOKBACK_PERIOD = "6mo"

# ---------------------------------------------------------------------------
# Arsitektur & training hyperparameters model UNIVERSAL
# ---------------------------------------------------------------------------
# Arsitektur berbeda dari model per-saham lama: dua layer LSTM terpisah
# (bukan nn.LSTM(num_layers=2)) diikuti fc1(64->16) + ReLU + fc2(16->1).
# Lihat src/model.py::UniversalLSTM -- HARUS SAMA PERSIS dengan arsitektur
# saat training agar load_state_dict berhasil.
HIDDEN_SIZE = 64
FC_HIDDEN_SIZE = 16
NUM_LAYERS = 2
DROPOUT = 0.2

EPOCHS = 150
PATIENCE = 15
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
SEED = 42


# ---------------------------------------------------------------------------
# Horizon prediksi multi-hari
# ---------------------------------------------------------------------------
# Model dilatih untuk memprediksi HANYA 1 hari ke depan. Untuk horizon > 1,
# dashboard melakukan forecasting rekursif: hasil prediksi hari ke-t dipakai
# sebagai bagian dari input untuk memprediksi hari ke-(t+1), dst. Error bisa
# terakumulasi semakin jauh horizonnya -- karena itu tiap horizon punya nilai
# MAPE (dari backtest) masing-masing yang ditampilkan ke pengguna.
#
# Nilai = jumlah HARI PERDAGANGAN (bukan hari kalender).
HORIZON_OPTIONS = {
    "1 Hari": 1,
    "3 Hari": 3,
    "1 Minggu": 7,
    "2 Minggu": 10,
    "1 Bulan": 22,
}
DEFAULT_HORIZON_LABEL = "1 Hari"

# Batas maksimum jumlah sample backtest per horizon (agar tetap responsif)
MAX_BACKTEST_SAMPLES = 300


def raw_data_path(stock_code: str) -> Path:
    return DATA_RAW_DIR / f"{stock_code}.csv"


def dynamic_scaler_path(stock_code: str) -> Path:
    """Path cache untuk scaler yang di-fit otomatis (saham di luar ~300
    saham yang scaler-nya sudah dibundel di scalers.pkl)."""
    return DYNAMIC_SCALERS_DIR / f"{stock_code}.pkl"


# ---------------------------------------------------------------------------
# Legacy: 3 model per-saham yang lama (TLKM, BYAN, ASII), sebelum dashboard
# beralih ke satu model universal. Dipertahankan hanya supaya src/train.py
# dan src/download_data.py (skrip re-training saham individual) tetap bisa
# jalan; TIDAK dipakai oleh dashboard/predict.py lagi.
LEGACY_TICKERS = {
    "TLKM": "TLKM.JK",
    "BYAN": "BYAN.JK",
    "ASII": "ASII.JK",
}
LEGACY_MODEL_FILENAMES = {code: f"best_lstm_{code}.pt" for code in LEGACY_TICKERS}
LEGACY_SCALER_FILENAMES = {code: f"scaler_{code}.pkl" for code in LEGACY_TICKERS}


def legacy_model_path(stock_code: str) -> Path:
    return MODELS_DIR / LEGACY_MODEL_FILENAMES[stock_code]


def legacy_scaler_path(stock_code: str) -> Path:
    return SCALERS_DIR / LEGACY_SCALER_FILENAMES[stock_code]
