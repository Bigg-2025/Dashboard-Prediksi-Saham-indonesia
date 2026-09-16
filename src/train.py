"""
Melatih ulang model LSTM untuk satu atau semua saham.

Dashboard TIDAK memanggil script ini -- ini hanya untuk (re-)training model
secara terpisah, offline. Setelah selesai, file .pt dan .pkl yang dihasilkan
otomatis tersimpan ke models/ dan scalers/ dan langsung terpakai oleh
dashboard pada run berikutnya.

Jalankan salah satu:
    python -m src.train                # latih semua saham di config.TICKERS
    python -m src.train TLKM BYAN       # latih saham tertentu saja
"""

import random
import sys

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from . import config, preprocess
from .model import LSTMRegressor


class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def set_seed(seed: int = config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_stock(stock_code: str, device: torch.device):
    print(f"\n{'=' * 60}\nTraining {stock_code}\n{'=' * 60}")

    df = preprocess.load_raw_csv(config.raw_data_path(stock_code))
    n = len(df)
    train_end = int(n * config.TRAIN_SPLIT)
    val_end = int(n * config.VAL_SPLIT)
    print(f"Train: {train_end} | Val: {val_end - train_end} | Test: {n - val_end} baris")

    train_raw = df.iloc[:train_end]
    scaler = preprocess.fit_scaler(train_raw)
    scaled_values = scaler.transform(df[config.FEATURE_COLS])

    X_all, y_all = preprocess.create_sequences(scaled_values, config.WINDOW_SIZE, config.TARGET_IDX)
    train_size = train_end - config.WINDOW_SIZE
    val_size = val_end - config.WINDOW_SIZE

    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val, y_val = X_all[train_size:val_size], y_all[train_size:val_size]
    X_test, y_test = X_all[val_size:], y_all[val_size:]

    train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)

    model = LSTMRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model_path = config.legacy_model_path(stock_code)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                running_val += loss.item() * xb.size(0)
        val_loss = running_val / len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping pada epoch {epoch} (val_loss terbaik: {best_val_loss:.6f})")
            break

    # Evaluasi memakai model terbaik
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def predict(loader):
        preds = []
        with torch.no_grad():
            for xb, _ in loader:
                xb = xb.to(device)
                preds.append(model(xb).cpu().numpy())
        return np.concatenate(preds).flatten()

    y_pred = preprocess.inverse_target(scaler, predict(test_loader))
    y_true = preprocess.inverse_target(scaler, y_test)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    print(f"RMSE: {rmse:,.2f} | MAE: {mae:,.2f} | MAPE: {mape:.2f}% | R2: {r2:.4f}")

    joblib.dump(scaler, config.legacy_scaler_path(stock_code))
    print(f"Model  -> {model_path}")
    print(f"Scaler -> {config.legacy_scaler_path(stock_code)}")


def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Menggunakan device: {device}")

    codes = sys.argv[1:] or list(config.LEGACY_TICKERS.keys())
    for code in codes:
        if code not in config.LEGACY_TICKERS:
            print(f"Lewati '{code}': tidak ada di config.TICKERS")
            continue
        train_one_stock(code, device)


if __name__ == "__main__":
    main()
