"""Helper kecil untuk format tampilan yang dipakai di banyak halaman dashboard."""

import pandas as pd


def format_rupiah(value: float) -> str:
    """Format angka jadi string Rupiah, mis. 2706.49 -> 'Rp 2.706,49'."""
    return f"Rp {value:,.2f}".replace(",", "@").replace(".", ",").replace("@", ".")


def format_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.2f}%"


def format_date_id(date: pd.Timestamp) -> str:
    """Format tanggal ke gaya Indonesia, mis. '28 Juli 2026'."""
    bulan_id = [
        "Januari", "Februari", "Maret", "April", "Mei", "Juni",
        "Juli", "Agustus", "September", "Oktober", "November", "Desember",
    ]
    date = pd.Timestamp(date)
    return f"{date.day} {bulan_id[date.month - 1]} {date.year}"
