"""
Fungsi preprocessing yang dipakai bersama oleh training (src/train.py) dan
inference (src/predict.py), supaya kedua alur selalu konsisten.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from . import config


def load_raw_csv(path) -> pd.DataFrame:
    """Baca CSV mentah, urutkan berdasarkan tanggal, buang kolom Adj Close."""
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])
    return df


def create_sequences(values: np.ndarray, window_size: int, target_idx: int):
    """Bentuk sliding-window sequence (X) dan target (y) dari array ternormalisasi."""
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size : i])
        y.append(values[i, target_idx])
    return np.array(X), np.array(y)


def fit_scaler(train_df: pd.DataFrame) -> MinMaxScaler:
    """Fit MinMaxScaler HANYA pada data train, agar tidak ada kebocoran info."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_df[config.FEATURE_COLS])
    return scaler


def inverse_target(scaler: MinMaxScaler, scaled_target: np.ndarray) -> np.ndarray:
    """Kembalikan nilai Close hasil prediksi (skala 0-1) ke skala harga asli.

    MinMaxScaler di-fit pada 5 kolom sekaligus, jadi untuk inverse_transform
    satu kolom target kita perlu "menyamarkan" array dummy berukuran penuh,
    isi kolom target dengan nilai yang mau di-inverse, lalu ambil kembali
    kolom itu setelah inverse_transform.
    """
    dummy = np.zeros((len(scaled_target), len(config.FEATURE_COLS)))
    dummy[:, config.TARGET_IDX] = scaled_target
    return scaler.inverse_transform(dummy)[:, config.TARGET_IDX]


def build_last_window(df: pd.DataFrame, scaler: MinMaxScaler, window_size: int = config.WINDOW_SIZE):
    """Ambil window_size baris TERAKHIR dari df, scale, dan bentuk jadi
    sequence siap-inference dengan shape (1, window_size, n_features).

    Dipakai saat inference: kita tidak butuh y, hanya X untuk memprediksi
    hari berikutnya setelah baris terakhir df.
    """
    if len(df) < window_size:
        raise ValueError(
            f"Data historis kurang dari {window_size} hari perdagangan "
            f"(hanya {len(df)} baris tersedia). Tidak bisa membentuk sequence."
        )
    recent = df[config.FEATURE_COLS].iloc[-window_size:]
    scaled = scaler.transform(recent)
    sequence = scaled.reshape(1, window_size, len(config.FEATURE_COLS))
    return sequence
