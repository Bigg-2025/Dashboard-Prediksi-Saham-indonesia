"""Helper tampilan yang dipakai bersama di semua halaman dashboard.

Gaya visual terinspirasi aplikasi trading saham (mis. Stockbit): bersih,
angka besar & tegas, hijau untuk naik dan merah untuk turun, minim
gradient/dekorasi berlebihan.
"""

from pathlib import Path

import streamlit as st

_CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def apply_theme():
    """Suntikkan CSS bersama. Panggil sekali di awal tiap halaman,
    setelah st.set_page_config()."""
    if _CSS_PATH.exists():
        st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str, badge: str = None):
    """Render header halaman: kartu putih bersih dengan aksen garis hijau
    di kiri (bukan gradient penuh), meniru header bersih ala aplikasi
    trading saham."""
    badge_html = f'<div class="page-badge">{badge}</div>' if badge else ""
    st.markdown(
        f"""
        <div class="page-header">
            {badge_html}
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def accuracy_pill(mape: float) -> str:
    """Beri label kualitatif + kelas CSS berdasarkan nilai MAPE (%)."""
    if mape < 5:
        return '<span class="pill pill-accuracy-good">Akurasi Tinggi</span>'
    if mape < 12:
        return '<span class="pill pill-accuracy-medium">Akurasi Sedang</span>'
    return '<span class="pill pill-accuracy-low">Akurasi Rendah</span>'


def change_pill_html(change_pct: float) -> str:
    """Pill kecil hijau/merah dengan panah, ala tampilan naik/turun harga saham."""
    arrow = "▲" if change_pct >= 0 else "▼"
    cls = "positive" if change_pct >= 0 else "negative"
    sign = "+" if change_pct >= 0 else ""
    return f'<span class="change-pill {cls}">{arrow} {sign}{change_pct:.2f}%</span>'


def quote_card(
    ticker: str,
    company_name: str,
    last_label: str,
    last_value_str: str,
    predicted_label: str,
    predicted_value_str: str,
    change_pct: float,
    meta_caption: str,
):
    """Kartu 'quote' ala halaman detail saham: ticker + nama perusahaan di
    atas, lalu harga terakhir -> panah -> harga prediksi berdampingan,
    lengkap dengan pill naik/turun. Dipakai di halaman Prediction supaya
    angka besar tidak pernah terpotong (ellipsis) seperti pada st.metric.
    """
    st.markdown(
        f"""
        <div class="quote-card">
            <div class="quote-top-row">
                <div>
                    <span class="quote-ticker">{ticker}</span>
                    <span class="quote-company">{company_name}</span>
                </div>
                <div class="quote-meta">{meta_caption}</div>
            </div>
            <div class="quote-grid">
                <div>
                    <div class="quote-block-label">{last_label}</div>
                    <div class="quote-block-value">{last_value_str}</div>
                </div>
                <div class="quote-arrow">&#8594;</div>
                <div>
                    <div class="quote-block-label">{predicted_label}</div>
                    <div class="quote-block-value">{predicted_value_str}</div>
                    {change_pill_html(change_pct)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
