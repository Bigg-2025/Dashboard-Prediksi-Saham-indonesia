"""Builder chart Plotly yang dipakai bersama oleh halaman Prediction & Historical Data."""

import pandas as pd
import plotly.graph_objects as go


def line_close_chart(df: pd.DataFrame, title: str = "Harga Penutupan") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"], y=df["Close"],
            mode="lines", name="Close",
            line=dict(color="#0E9F6E", width=2),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Harga (Rp)",
        template="plotly_white",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def forecast_path_chart(history: pd.DataFrame, forecast_dates, forecast_closes,
                         lookback: int = 90) -> go.Figure:
    """Grafik historis + lintasan prediksi multi-hari, warna hijau/merah
    mengikuti arah prediksi (naik/turun) ala aplikasi trading saham."""
    recent = history.tail(lookback)
    last_date = recent["Date"].iloc[-1]
    last_close = recent["Close"].iloc[-1]

    is_up = forecast_closes[-1] >= last_close
    forecast_color = "#16A34A" if is_up else "#DC2626"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recent["Date"], y=recent["Close"],
            mode="lines", name="Harga Historis",
            line=dict(color="#374151", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[last_date] + list(forecast_dates),
            y=[last_close] + list(forecast_closes),
            mode="lines+markers", name="Prediksi",
            line=dict(color=forecast_color, width=2.5, dash="dash"),
            marker=dict(size=7, color=forecast_color),
        )
    )
    # Tandai titik prediksi terakhir (horizon target) secara khusus
    fig.add_trace(
        go.Scatter(
            x=[forecast_dates[-1]], y=[forecast_closes[-1]],
            mode="markers", name="Target Horizon",
            marker=dict(size=14, color=forecast_color, symbol="star",
                        line=dict(width=1.5, color="white")),
            showlegend=False,
        )
    )
    fig.update_layout(
        title="Harga Historis & Lintasan Prediksi",
        xaxis_title="Tanggal",
        yaxis_title="Harga (Rp)",
        template="plotly_white",
        height=440,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def candlestick_chart(df: pd.DataFrame, title: str = "Candlestick") -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Date"],
                open=df["Open"], high=df["High"],
                low=df["Low"], close=df["Close"],
                increasing_line_color="#16a34a",
                decreasing_line_color="#dc2626",
                name="OHLC",
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Harga (Rp)",
        template="plotly_white",
        height=480,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_rangeslider_visible=False,
    )
    return fig


def mape_by_horizon_chart(labels, mape_values, active_label: str = None) -> go.Figure:
    """Bar chart perbandingan MAPE (%) di semua pilihan horizon, dengan
    horizon yang sedang aktif disorot warna berbeda."""
    colors = ["#0E9F6E" if lbl == active_label else "#D1FAE5" for lbl in labels]
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(labels), y=list(mape_values),
                marker_color=colors,
                text=[f"{v:.2f}%" for v in mape_values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="MAPE (Test Set) per Horizon",
        xaxis_title="Horizon",
        yaxis_title="MAPE (%)",
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=10, t=50, b=10),
        showlegend=False,
    )
    return fig


def volume_chart(df: pd.DataFrame, title: str = "Volume Perdagangan") -> go.Figure:
    colors = [
        "#16a34a" if c >= o else "#dc2626"
        for o, c in zip(df["Open"], df["Close"])
    ]
    fig = go.Figure(data=[go.Bar(x=df["Date"], y=df["Volume"], marker_color=colors)])
    fig.update_layout(
        title=title,
        xaxis_title="Tanggal",
        yaxis_title="Volume",
        template="plotly_white",
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
