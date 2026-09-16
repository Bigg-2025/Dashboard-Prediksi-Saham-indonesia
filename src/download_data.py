"""
Unduh data historis 5 tahun terakhir untuk semua saham dari Yahoo Finance
dan simpan ke data/raw/<KODE>.csv.

Script ini HANYA dipakai untuk (re-)training. Dashboard tidak memanggil
script ini -- dashboard mengambil data terbaru langsung saat halaman dibuka
(lihat src/predict.py -> fetch_latest_data()).

Jalankan:
    python -m src.download_data
"""

import sys

import yfinance as yf

from . import config


def download_stock(stock_code: str, ticker: str, period: str = config.TRAIN_PERIOD):
    print(f"[download_data] Mengunduh {stock_code} ({ticker}), periode={period} ...")
    df = yf.download(ticker, period=period, auto_adjust=False, progress=False)

    if df.empty:
        raise RuntimeError(
            f"Tidak ada data yang diterima untuk {ticker}. "
            "Periksa koneksi internet atau kode ticker."
        )

    # yfinance versi terbaru mengembalikan MultiIndex kolom saat men-download
    # satu ticker sekaligus; ratakan supaya konsisten dengan notebook training.
    if isinstance(df.columns, __import__("pandas").MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()  # kolom "Date" jadi kolom biasa
    out_path = config.raw_data_path(stock_code)
    df.to_csv(out_path, index=False)
    print(f"[download_data] Tersimpan -> {out_path} ({len(df)} baris)")


def main():
    codes = sys.argv[1:] or list(config.LEGACY_TICKERS.keys())
    for code in codes:
        if code not in config.LEGACY_TICKERS:
            print(f"[download_data] Lewati '{code}': tidak ada di config.TICKERS")
            continue
        download_stock(code, config.LEGACY_TICKERS[code])


if __name__ == "__main__":
    main()
