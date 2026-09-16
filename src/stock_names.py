"""
Nama lengkap perusahaan (untuk ditampilkan di dashboard), diambil on-demand
dari Yahoo Finance dan di-cache ke disk supaya tidak perlu request ulang
setiap kali halaman dibuka -- terutama penting sekarang karena ada ~300
saham yang didukung, bukan 3.
"""

import json

import yfinance as yf

from . import config

_CACHE_PATH = config.DATA_PROCESSED_DIR / "stock_names_cache.json"


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # noqa: BLE001
        # Cache gagal ditulis (mis. filesystem read-only) -- tidak fatal,
        # nama tetap bisa dipakai untuk request saat ini, hanya tidak
        # tersimpan untuk request berikutnya.
        pass


def get_stock_name(stock_code: str) -> str:
    """Ambil nama perusahaan untuk satu kode saham.

    Prioritas: (1) data/idx_stock_list.csv yang dibundel bersama proyek
    (offline, instan, mencakup ~950 saham IDX -- lihat config.STOCK_NAMES),
    (2) fallback ke Yahoo Finance untuk kode yang tidak ada di daftar itu
    (dengan cache disk), (3) kode saham itu sendiri jika semua gagal.
    """
    bundled_name = config.STOCK_NAMES.get(stock_code)
    if bundled_name:
        return bundled_name

    cache = _load_cache()
    if stock_code in cache:
        return cache[stock_code]

    ticker = config.TICKERS.get(stock_code)
    name = stock_code
    if ticker:
        try:
            info = yf.Ticker(ticker).info
            name = info.get("longName") or info.get("shortName") or stock_code
        except Exception:  # noqa: BLE001
            name = stock_code

    cache[stock_code] = name
    _save_cache(cache)
    return name
