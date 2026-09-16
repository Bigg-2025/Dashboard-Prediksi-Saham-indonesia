"""
Arsitektur model LSTM.

PENTING: kelas-kelas ini harus sama persis dengan arsitektur yang dipakai
saat melatih file model di folder models/. Jika arsitektur di sini diubah,
model lama tidak akan bisa di-load lagi via load_state_dict.
"""

import torch
import torch.nn as nn

from . import config


class UniversalLSTM(nn.Module):
    """Model LSTM universal, dilatih atas banyak saham (~300) sekaligus.

    Arsitektur: dua layer LSTM terpisah (lstm1 -> lstm2, masing-masing
    single-layer) diikuti fc1(hidden->fc_hidden) + ReLU + Dropout +
    fc2(fc_hidden->1). Nama-nama sub-modul (lstm1, lstm2, fc1, fc2) harus
    sama persis dengan checkpoint universal_lstm_best.pth.
    """

    def __init__(
        self,
        input_size: int = len(config.FEATURE_COLS),
        hidden_size: int = config.HIDDEN_SIZE,
        fc_hidden_size: int = config.FC_HIDDEN_SIZE,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        last_step = out[:, -1, :]
        last_step = self.dropout(last_step)
        hidden = self.relu(self.fc1(last_step))
        return self.fc2(hidden)


def load_universal_model(device: torch.device = None) -> UniversalLSTM:
    """Muat model LSTM universal (satu model untuk semua saham)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniversalLSTM().to(device)
    state_dict = torch.load(config.UNIVERSAL_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


class LSTMRegressor(nn.Module):
    """LSTM per-saham (legacy, sebelum ada model universal)."""

    def __init__(
        self,
        input_size: int = len(config.FEATURE_COLS),
        hidden_size: int = config.HIDDEN_SIZE,
        num_layers: int = config.NUM_LAYERS,
        dropout: float = config.DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.fc(last_step)


def load_model(stock_code: str, device: torch.device = None) -> LSTMRegressor:
    """Muat model LSTM per-saham legacy untuk satu kode saham."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMRegressor().to(device)
    state_dict = torch.load(config.legacy_model_path(stock_code), map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model
