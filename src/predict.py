"""
Modul inference yang dipakai langsung oleh dashboard Streamlit.

Alur untuk prediksi 1 hari:
    Yahoo Finance -> download data terbaru -> preprocessing -> load scaler
    -> load model -> prediksi -> (dashboard) visualisasi

Untuk horizon > 1 hari, model (yang hanya dilatih memprediksi 1 hari ke
depan) dipakai secara REKURSIF: prediksi Close hari ke-t dipakai sebagai
bagian dari input sequence untuk memprediksi hari ke-(t+1), dan seterusnya.
Karena Open/High/Low/Volume hari ke-t tidak diketahui, dipakai pendekatan
umum: Open=High=Low=Close=harga prediksi, Volume=volume terakhir yang
diketahui. Konsekuensinya, error bisa terakumulasi semakin jauh horizonnya
-- karena itu setiap horizon dievaluasi ulang lewat backtest (lihat
`backtest_mape`) supaya pengguna tahu seberapa reliabel prediksi tersebut.

Dashboard TIDAK melakukan training di sini -- hanya memuat model & scaler
yang sudah dilatih sebelumnya (src/train.py) lalu melakukan forward pass.
"""

from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf

from . import config, preprocess
from .model import load_universal_model

OHLC_IDX = [config.FEATURE_COLS.index(c) for c in ("Open", "High", "Low", "Close")]
VOLUME_IDX = config.FEATURE_COLS.index("Volume")


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_latest_data(stock_code: str, period: str = config.INFERENCE_LOOKBACK_PERIOD) -> pd.DataFrame:
    """Unduh data OHLCV terbaru dari Yahoo Finance untuk satu saham.

    auto_adjust=False dipakai supaya kolom "Close" konsisten dengan apa yang
    dipakai saat training (lihat notebooks/*.ipynb), bukan versi adjusted.
    """
    ticker = config.TICKERS[stock_code]
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise RuntimeError(
            f"Tidak menerima data dari Yahoo Finance untuk {ticker}. "
            "Coba lagi beberapa saat, atau periksa koneksi internet."
        )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    if "Adj Close" in df.columns:
        df = df.drop(columns=["Adj Close"])

    df = df.sort_values("Date").reset_index(drop=True)
    return df[["Date"] + config.FEATURE_COLS]


def next_trading_date(last_date: pd.Timestamp) -> pd.Timestamp:
    """Perkiraan tanggal perdagangan berikutnya (melompati Sabtu & Minggu).

    Estimasi sederhana, tidak memperhitungkan hari libur nasional / bursa
    efek Indonesia secara spesifik.
    """
    next_date = last_date + pd.Timedelta(days=1)
    while next_date.weekday() >= 5:  # 5=Sabtu, 6=Minggu
        next_date += pd.Timedelta(days=1)
    return next_date


def next_n_trading_dates(last_date: pd.Timestamp, n: int) -> list:
    dates = []
    current = last_date
    for _ in range(n):
        current = next_trading_date(current)
        dates.append(current)
    return dates


# ---------------------------------------------------------------------------
# Model / scaler cache
# ---------------------------------------------------------------------------
# Satu model LSTM universal dipakai untuk SEMUA saham -- jadi cukup dimuat
# sekali. Scaler tetap per-saham (karakteristik harga tiap saham beda jauh),
# tapi semuanya dibaca sekali dari scalers.pkl lalu disimpan di memori.
_model_cache: dict = {}
_scalers_dict_cache: dict = None


def _get_model(stock_code: str = None):
    """Ambil model LSTM universal (parameter stock_code diabaikan, hanya
    dipertahankan supaya pemanggilnya tetap seragam)."""
    if "universal" not in _model_cache:
        _model_cache["universal"] = load_universal_model()
    return _model_cache["universal"]


def _get_all_scalers() -> dict:
    global _scalers_dict_cache
    if _scalers_dict_cache is None:
        _scalers_dict_cache = joblib.load(config.SCALERS_PATH)
    return _scalers_dict_cache


_dynamic_scaler_cache: dict = {}


def _get_or_fit_dynamic_scaler(stock_code: str):
    """Untuk saham DI LUAR ~300 saham yang dipakai melatih model universal
    (tidak ada di scalers.pkl): fit MinMaxScaler sendiri dari data historis
    saham tersebut, memakai metodologi yang SAMA seperti training (fit
    hanya pada 70% data pertama / train split, lihat preprocess.fit_scaler
    & src/train.py) supaya konsisten dan tidak ada kebocoran data dari
    test set.

    Hasil fit di-cache ke disk (scalers/dynamic/<KODE>.pkl) supaya hanya
    perlu mengunduh & fit sekali per saham, bukan setiap request.
    """
    if stock_code in _dynamic_scaler_cache:
        return _dynamic_scaler_cache[stock_code]

    cache_path = config.dynamic_scaler_path(stock_code)
    if cache_path.exists():
        scaler = joblib.load(cache_path)
        _dynamic_scaler_cache[stock_code] = scaler
        return scaler

    history = _get_backtest_history(stock_code)  # unduh + cache 5y data
    train_end = int(len(history) * config.TRAIN_SPLIT)
    if train_end < config.WINDOW_SIZE:
        raise ValueError(
            f"Data historis '{stock_code}' terlalu sedikit untuk membentuk "
            f"scaler yang andal (butuh minimal {config.WINDOW_SIZE} hari "
            "perdagangan pada porsi data train)."
        )
    scaler = preprocess.fit_scaler(history.iloc[:train_end])

    joblib.dump(scaler, cache_path)
    _dynamic_scaler_cache[stock_code] = scaler
    return scaler


def _get_scaler(stock_code: str):
    scalers = _get_all_scalers()
    if stock_code in scalers:
        return scalers[stock_code]
    # Saham di luar ~300 saham pretrained -- kalibrasi scaler otomatis.
    return _get_or_fit_dynamic_scaler(stock_code)


def is_pretrained_stock(stock_code: str) -> bool:
    """True jika saham termasuk ~300 saham yang datanya dipakai melatih
    model universal (scaler pretrained, paling bisa diandalkan). False
    berarti scaler dikalibrasi otomatis saat pertama diminta."""
    return stock_code in config.PRETRAINED_SCALER_CODES


# ---------------------------------------------------------------------------
# Core rekursif multi-step forecast (batched, jalan untuk N sequence sekaligus)
# ---------------------------------------------------------------------------
def recursive_forecast_batch(model, initial_sequences: np.ndarray, steps: int) -> np.ndarray:
    """Forecast `steps` hari ke depan secara rekursif, untuk banyak sequence sekaligus.

    Parameters
    ----------
    initial_sequences: array shape (batch, window_size, n_features), SUDAH di-scale (0-1).
    steps: jumlah hari ke depan yang mau diprediksi.

    Returns
    -------
    array shape (batch, steps) berisi prediksi Close (masih dalam skala 0-1),
    satu kolom per hari ke depan.
    """
    seq = torch.tensor(initial_sequences, dtype=torch.float32)
    preds = []

    with torch.no_grad():
        for _ in range(steps):
            out = model(seq).squeeze(-1)  # (batch,) scaled predicted close
            preds.append(out)

            last_row = seq[:, -1, :].clone()
            new_row = last_row.clone()
            new_row[:, OHLC_IDX] = out.unsqueeze(1)  # Open=High=Low=Close=prediksi
            # Volume: pertahankan nilai terakhir yang diketahui (asumsi sederhana)
            new_row[:, VOLUME_IDX] = last_row[:, VOLUME_IDX]

            seq = torch.cat([seq[:, 1:, :], new_row.unsqueeze(1)], dim=1)

    return torch.stack(preds, dim=1).cpu().numpy()  # (batch, steps)


# ---------------------------------------------------------------------------
# Live forecast (dipanggil dashboard)
# ---------------------------------------------------------------------------
@dataclass
class ForecastResult:
    stock_code: str
    ticker: str
    horizon_days: int
    history: pd.DataFrame
    last_date: pd.Timestamp
    last_close: float
    forecast_dates: list = field(default_factory=list)
    forecast_closes: list = field(default_factory=list)
    mape: float = None
    rmse: float = None
    n_backtest_samples: int = None

    @property
    def predicted_date(self):
        return self.forecast_dates[-1]

    @property
    def predicted_close(self) -> float:
        return self.forecast_closes[-1]

    @property
    def change_value(self) -> float:
        return self.predicted_close - self.last_close

    @property
    def change_pct(self) -> float:
        return (self.change_value / self.last_close) * 100

    def as_path_df(self) -> pd.DataFrame:
        """Dataframe (Date, Close) berisi seluruh lintasan prediksi hari-per-hari."""
        return pd.DataFrame({"Date": self.forecast_dates, "Close": self.forecast_closes})


def forecast(stock_code: str, horizon_days: int, history: pd.DataFrame = None,
             with_mape: bool = True) -> ForecastResult:
    """Prediksi harga Close untuk `horizon_days` hari perdagangan ke depan.

    Parameters
    ----------
    stock_code: kode saham, mis. "TLKM" (harus ada di config.TICKERS).
    horizon_days: jumlah hari perdagangan ke depan yang diprediksi (>=1).
    history: opsional, data historis yang sudah diambil sebelumnya (supaya
        tidak mengunduh dua kali untuk halaman yang sama).
    with_mape: jika True, sertakan hasil backtest MAPE untuk horizon ini.
    """
    if stock_code not in config.TICKERS:
        raise ValueError(f"Kode saham '{stock_code}' tidak dikenal.")
    if horizon_days < 1:
        raise ValueError("horizon_days harus >= 1")

    if history is None:
        history = fetch_latest_data(stock_code)

    model = _get_model(stock_code)
    scaler = _get_scaler(stock_code)

    sequence = preprocess.build_last_window(history, scaler, config.WINDOW_SIZE)  # (1, window, feat)
    scaled_preds = recursive_forecast_batch(model, sequence, horizon_days)[0]  # (horizon,)
    predicted_closes = preprocess.inverse_target(scaler, scaled_preds).tolist()

    last_row = history.iloc[-1]
    last_date = pd.Timestamp(last_row["Date"])
    last_close = float(last_row["Close"])
    forecast_dates = next_n_trading_dates(last_date, horizon_days)

    result = ForecastResult(
        stock_code=stock_code,
        ticker=config.TICKERS[stock_code],
        horizon_days=horizon_days,
        history=history,
        last_date=last_date,
        last_close=last_close,
        forecast_dates=forecast_dates,
        forecast_closes=predicted_closes,
    )

    if with_mape:
        try:
            metrics = backtest_mape(stock_code, horizon_days)
            result.mape = metrics["mape"]
            result.rmse = metrics["rmse"]
            result.n_backtest_samples = metrics["n_samples"]
        except Exception:  # noqa: BLE001
            # Backtest gagal (mis. data historis lokal tidak lengkap) --
            # prediksi tetap ditampilkan, hanya tanpa metrik MAPE.
            pass

    return result


def _get_backtest_history(stock_code: str) -> pd.DataFrame:
    """Data historis panjang (`config.TRAIN_PERIOD`) untuk backtest, dengan
    cache file lokal di data/raw/<KODE>.csv supaya tidak perlu mengunduh
    ulang dari Yahoo Finance setiap kali backtest dijalankan."""
    cache_path = config.raw_data_path(stock_code)
    if cache_path.exists():
        return preprocess.load_raw_csv(cache_path)

    df = fetch_latest_data(stock_code, period=config.TRAIN_PERIOD)
    df.to_csv(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Backtest: evaluasi akurasi (MAPE/RMSE) model per horizon pada test set
# ---------------------------------------------------------------------------
def backtest_mape(stock_code: str, horizon_days: int, max_samples: int = config.MAX_BACKTEST_SAMPLES) -> dict:
    """Hitung MAPE & RMSE model untuk horizon tertentu, dievaluasi pada data
    TEST SET (15% data terakhir, tidak pernah dipakai untuk training/fit
    scaler) memakai forecasting rekursif yang sama seperti `forecast()`.

    Karena model universal mendukung ~300 saham, data historis TIDAK
    dibundel lokal per saham (tidak seperti model per-saham lama) --
    dashboard mengunduh data `config.TRAIN_PERIOD` dari Yahoo Finance
    (dengan cache file lokal di data/raw/) lalu memakai 15% data terakhir
    sebagai test set, konsisten dengan pembagian train/val/test saat
    training. Butuh koneksi internet pada pemanggilan pertama untuk tiap
    saham; hasil unduhan berikutnya dipakai dari cache file lokal.
    """
    df = _get_backtest_history(stock_code)
    n = len(df)

    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * config.VAL_SPLIT)

    scaler = _get_scaler(stock_code)
    model = _get_model(stock_code)

    scaled_values = scaler.transform(df[config.FEATURE_COLS])
    actual_close = df["Close"].to_numpy()

    window = config.WINDOW_SIZE
    # start index t = "hari terakhir yang diketahui" sebelum forecast dimulai.
    # butuh: cukup histori di belakang (t >= window-1) DAN cukup masa depan
    # asli untuk dibandingkan (t + horizon_days <= n-1), serta t ada di
    # rentang test set (t >= val_end).
    start_min = max(val_end, window - 1)
    start_max = n - 1 - horizon_days
    candidate_starts = list(range(start_min, start_max + 1))

    if len(candidate_starts) == 0:
        raise ValueError(
            f"Tidak cukup data test set untuk backtest horizon={horizon_days} hari."
        )

    if len(candidate_starts) > max_samples:
        # Ambil sample merata di sepanjang test set, bukan hanya bagian awal,
        # supaya representatif tanpa membuat backtest jadi lambat.
        idx = np.linspace(0, len(candidate_starts) - 1, max_samples).astype(int)
        candidate_starts = [candidate_starts[i] for i in idx]

    batch_sequences = np.stack(
        [scaled_values[t - window + 1 : t + 1] for t in candidate_starts]
    )  # (batch, window, features)

    scaled_preds = recursive_forecast_batch(model, batch_sequences, horizon_days)  # (batch, horizon)
    final_scaled_pred = scaled_preds[:, -1]
    predicted_close = preprocess.inverse_target(scaler, final_scaled_pred)

    actual_future = np.array([actual_close[t + horizon_days] for t in candidate_starts])

    ape = np.abs((actual_future - predicted_close) / actual_future) * 100
    mape = float(np.mean(ape))
    rmse = float(np.sqrt(np.mean((actual_future - predicted_close) ** 2)))

    return {"mape": mape, "rmse": rmse, "n_samples": len(candidate_starts)}


def backtest_all_horizons(stock_code: str, horizon_options: dict = None) -> dict:
    """Jalankan backtest_mape untuk semua opsi horizon di config.HORIZON_OPTIONS
    sekaligus. Berguna untuk menampilkan grafik perbandingan MAPE per horizon.

    Returns: {label: {"mape": ..., "rmse": ..., "n_samples": ...}, ...}
    """
    horizon_options = horizon_options or config.HORIZON_OPTIONS
    results = {}
    for label, h in horizon_options.items():
        try:
            results[label] = backtest_mape(stock_code, h)
        except ValueError:
            results[label] = None
    return results
