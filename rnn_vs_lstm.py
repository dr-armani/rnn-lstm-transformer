"""
# SETUP (run once in your terminal before executing this script)
#
# 1. Create and activate a virtual environment:
#      python3 -m venv venv
#      source venv/bin/activate          # macOS / Linux
#      venv\Scripts\activate             # Windows
#
# 2. Install dependencies:
#      pip install torch numpy pandas scikit-learn matplotlib seaborn yfinance scipy
#
# 3. Run the script:
#      python rnn_vs_lstm_demo.py
#
# Runtime: ~5 min on Apple Silicon (MPS), ~15 min on CPU.
# Output:  7 PNG plots saved to the current directory + a summary table in the terminal.
"""

from __future__ import annotations

import random
import warnings
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yfinance as yf
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# =========================
# 1) Global Configuration
# =========================
TICKERS = ["AAPL", "MSFT", "GOOGL"]
SEQ_LEN = 60
BATCH_SIZE = 32
EPOCHS = 150
LR = 0.001
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
INITIAL_CAPITAL = 1000.0
SEED = 1


# =========================
# Seeds and Device
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(seed: int) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


set_seed(SEED)
DEVICE = get_device(SEED)


# =========================
# 2) Data Engineering
# =========================
def download_data() -> dict[str, pd.DataFrame]:
    data_dict: dict[str, pd.DataFrame] = {}
    end_date = datetime.today()
    start_date = end_date - timedelta(days=365 * 4 + 30)

    for ticker in TICKERS:
        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                auto_adjust=False,
                progress=False,
            )
            if df.empty:
                print(f"[WARN] No data returned for {ticker}. Skipping.")
                continue
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            data_dict[ticker] = df
        except Exception as exc:
            print(f"[WARN] Failed to download {ticker}: {exc}. Skipping.")
    return data_dict


def prepare_data(
    df: pd.DataFrame, seq_len: int, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    required_cols = [price_col, "High", "Low", "Volume"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cleaned = df[required_cols].dropna().copy()
    cleaned.columns = ["Adj Close", "High", "Low", "Volume"]

    price_scaler = MinMaxScaler(feature_range=(0, 1))
    vol_scaler = MinMaxScaler(feature_range=(0, 1))

    price_scaled = price_scaler.fit_transform(cleaned[["Adj Close", "High", "Low"]].values)
    vol_scaled = vol_scaler.fit_transform(cleaned[["Volume"]].values)
    scaled_data = np.hstack([price_scaled, vol_scaled]).astype(np.float32)

    split_idx = int(len(scaled_data) * (1 - test_size))
    if split_idx <= seq_len or len(scaled_data) - split_idx <= 1:
        raise ValueError("Not enough samples after split for the requested sequence length.")

    train = scaled_data[:split_idx]
    test = scaled_data[split_idx - seq_len :]
    return train, test, price_scaler, vol_scaler


class StockDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data: np.ndarray, seq_len: int) -> None:
        self.data = data
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len, 0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([y], dtype=torch.float32)


# =========================
# 3) Model Architectures
# =========================
class RNNModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


# =========================
# 4) Training
# =========================
def train_model(
    rnn: RNNModel,
    lstm: LSTMModel,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[list[float], list[float], list[float], list[float]]:
    criterion = nn.MSELoss()
    rnn_opt = torch.optim.Adam(rnn.parameters(), lr=lr)
    lstm_opt = torch.optim.Adam(lstm.parameters(), lr=lr)

    rnn_train_losses: list[float] = []
    rnn_val_losses: list[float] = []
    lstm_train_losses: list[float] = []
    lstm_val_losses: list[float] = []

    for epoch in range(epochs):
        rnn.train()
        lstm.train()
        running_rnn_train = 0.0
        running_lstm_train = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            rnn_opt.zero_grad()
            rnn_pred = rnn(x_batch)
            rnn_loss = criterion(rnn_pred, y_batch)
            rnn_loss.backward()
            rnn_opt.step()
            running_rnn_train += rnn_loss.item() * x_batch.size(0)

            lstm_opt.zero_grad()
            lstm_pred = lstm(x_batch)
            lstm_loss = criterion(lstm_pred, y_batch)
            lstm_loss.backward()
            lstm_opt.step()
            running_lstm_train += lstm_loss.item() * x_batch.size(0)

        rnn_epoch_train = running_rnn_train / max(1, len(train_loader.dataset))
        lstm_epoch_train = running_lstm_train / max(1, len(train_loader.dataset))
        rnn_train_losses.append(rnn_epoch_train)
        lstm_train_losses.append(lstm_epoch_train)

        rnn.eval()
        lstm.eval()
        running_rnn_val = 0.0
        running_lstm_val = 0.0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)

                rnn_val_pred = rnn(x_val)
                lstm_val_pred = lstm(x_val)

                rnn_vloss = criterion(rnn_val_pred, y_val)
                lstm_vloss = criterion(lstm_val_pred, y_val)

                running_rnn_val += rnn_vloss.item() * x_val.size(0)
                running_lstm_val += lstm_vloss.item() * x_val.size(0)

        rnn_epoch_val = running_rnn_val / max(1, len(val_loader.dataset))
        lstm_epoch_val = running_lstm_val / max(1, len(val_loader.dataset))
        rnn_val_losses.append(rnn_epoch_val)
        lstm_val_losses.append(lstm_epoch_val)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1:>3}/{epochs}]  "
                f"RNN  — Train: {rnn_epoch_train:.4f} | Val: {rnn_epoch_val:.4f}   "
                f"LSTM — Train: {lstm_epoch_train:.4f} | Val: {lstm_epoch_val:.4f}"
            )

    return rnn_train_losses, rnn_val_losses, lstm_train_losses, lstm_val_losses


# =========================
# 5) Evaluation & Forecasting
# =========================
def inverse_target_from_price_scaler(
    scaled_values: np.ndarray, price_scaler: MinMaxScaler
) -> np.ndarray:
    dummy = np.zeros((len(scaled_values), price_scaler.n_features_in_), dtype=np.float32)
    dummy[:, 0] = scaled_values.reshape(-1)
    inv = price_scaler.inverse_transform(dummy)
    return inv[:, 0]


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    price_scaler: MinMaxScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    preds_scaled: list[float] = []
    actuals_scaled: list[float] = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_hat = model(x_batch).squeeze(-1).detach().cpu().numpy()
            y_true = y_batch.squeeze(-1).detach().cpu().numpy()
            preds_scaled.extend(y_hat.tolist())
            actuals_scaled.extend(y_true.tolist())

    preds_usd = inverse_target_from_price_scaler(np.asarray(preds_scaled), price_scaler)
    actuals_usd = inverse_target_from_price_scaler(np.asarray(actuals_scaled), price_scaler)

    rmse = float(np.sqrt(mean_squared_error(actuals_usd, preds_usd)))
    mae = float(mean_absolute_error(actuals_usd, preds_usd))
    mape = float(np.mean(np.abs((actuals_usd - preds_usd) / np.clip(actuals_usd, 1e-8, None))) * 100)

    metrics = {"RMSE": rmse, "MAE": mae, "MAPE": mape}
    return preds_usd, actuals_usd, metrics


def predict_future(
    model: nn.Module,
    last_sequence: np.ndarray,
    price_scaler: MinMaxScaler,
    steps: int = 30,
    device: torch.device = DEVICE,
) -> list[float]:
    model.eval()
    window = last_sequence.copy().astype(np.float32)
    forecasts_usd: list[float] = []

    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(window[np.newaxis, :, :], dtype=torch.float32, device=device)
            next_scaled = float(model(x).item())

            next_usd = float(
                inverse_target_from_price_scaler(np.asarray([next_scaled], dtype=np.float32), price_scaler)[
                    0
                ]
            )
            forecasts_usd.append(next_usd)

            next_row = window[-1].copy()
            next_row[0] = next_scaled
            window = np.vstack([window[1:], next_row])

    return forecasts_usd


# =========================
# 6) Input Sensitivity
# =========================
def compute_sensitivity(
    model: nn.Module, last_sequence: np.ndarray, device: torch.device = DEVICE
) -> np.ndarray:
    model.eval()
    x = torch.tensor(last_sequence[np.newaxis, :, :], dtype=torch.float32, device=device)
    x.requires_grad_(True)

    output = model(x).squeeze()
    model.zero_grad(set_to_none=True)
    output.backward()

    grad = x.grad.detach().cpu().numpy()[0]
    sens = np.mean(np.abs(grad), axis=1)
    max_val = float(np.max(sens))
    if max_val > 0:
        sens = sens / max_val
    return sens.astype(np.float32)


# =========================
# 7) Portfolio Simulation
# =========================
def simulate_portfolio(
    all_preds: dict[str, np.ndarray],
    all_actuals: dict[str, np.ndarray],
    n_days: int,
    initial_capital: float = INITIAL_CAPITAL,
) -> tuple[np.ndarray, list[str], int]:
    tickers = list(all_preds.keys())
    capital = initial_capital
    portfolio_values = [capital]
    holdings: list[str] = []
    switches = 0
    prev_holding = "Cash"

    for day in range(n_days):
        best_ticker = "Cash"
        best_return = -np.inf

        for ticker in tickers:
            today_actual = float(all_actuals[ticker][day])
            pred_today = float(all_preds[ticker][day])
            pred_return = (pred_today - today_actual) / max(today_actual, 1e-8)
            if pred_return > best_return:
                best_return = pred_return
                best_ticker = ticker

        if best_return <= 0:
            chosen = "Cash"
        else:
            chosen = best_ticker

        if chosen != prev_holding:
            switches += 1
        prev_holding = chosen
        holdings.append(chosen)

        if chosen == "Cash":
            portfolio_values.append(capital)
            continue

        price_today = float(all_actuals[chosen][day])
        price_next = float(all_actuals[chosen][day + 1])
        daily_return = (price_next - price_today) / max(price_today, 1e-8)
        capital *= 1 + daily_return
        portfolio_values.append(capital)

    return np.asarray(portfolio_values, dtype=np.float64), holdings, switches


def compute_bah_portfolio(
    all_actuals: dict[str, np.ndarray], n_days: int, initial_capital: float = INITIAL_CAPITAL
) -> np.ndarray:
    tickers = list(all_actuals.keys())
    per_stock = initial_capital / len(tickers)
    first_day_prices = {t: float(all_actuals[t][0]) for t in tickers}
    shares = {t: per_stock / max(first_day_prices[t], 1e-8) for t in tickers}

    values: list[float] = []
    for day in range(n_days + 1):
        total_val = 0.0
        for t in tickers:
            total_val += shares[t] * float(all_actuals[t][day])
        values.append(total_val)
    return np.asarray(values, dtype=np.float64)


# =========================
# 8) Plots
# =========================
def plot_loss_curves(
    ticker: str,
    rnn_train: list[float],
    rnn_val: list[float],
    lstm_train: list[float],
    lstm_val: list[float],
) -> None:
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(rnn_train) + 1)
    plt.plot(epochs, rnn_train, color="tab:blue", linestyle="-", label="RNN Train")
    plt.plot(epochs, rnn_val, color="tab:blue", linestyle="--", label="RNN Val")
    plt.plot(epochs, lstm_train, color="tab:orange", linestyle="-", label="LSTM Train")
    plt.plot(epochs, lstm_val, color="tab:orange", linestyle="--", label="LSTM Val")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss (log scale)")
    plt.title(f"Training vs Validation Loss — {ticker}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plot_01_loss_curves_{ticker}.png", dpi=150)
    plt.close()


def plot_test_predictions(
    tickers: list[str],
    actuals: dict[str, np.ndarray],
    rnn_preds: dict[str, np.ndarray],
    lstm_preds: dict[str, np.ndarray],
    metrics: dict[str, dict[str, dict[str, float]]],
) -> None:
    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 12), sharex=False)
    if len(tickers) == 1:
        axes = [axes]

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        y_true = actuals[ticker]
        y_rnn = rnn_preds[ticker]
        y_lstm = lstm_preds[ticker]
        x = np.arange(len(y_true))

        ax.plot(x, y_true, color="black", label="Actual", linewidth=1.8)
        ax.plot(x, y_rnn, color="tab:blue", alpha=0.7, label="RNN")
        ax.plot(x, y_lstm, color="tab:orange", alpha=0.7, label="LSTM")
        ymin = np.minimum(y_rnn, y_lstm)
        ymax = np.maximum(y_rnn, y_lstm)
        ax.fill_between(x, ymin, ymax, color="purple", alpha=0.15, label="Disagreement Zone")

        r_rmse = metrics[ticker]["RNN"]["RMSE"]
        l_rmse = metrics[ticker]["LSTM"]["RMSE"]
        ax.set_title(f"{ticker} — RMSE (RNN: {r_rmse:.2f}, LSTM: {l_rmse:.2f})")
        ax.set_ylabel("USD")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc="upper left", ncol=4, fontsize=9)

    axes[-1].set_xlabel("Test timestep")
    plt.tight_layout()
    plt.savefig("plot_02_test_predictions.png", dpi=150)
    plt.close()


def plot_forecast(
    tickers: list[str],
    recent_actuals: dict[str, np.ndarray],
    rnn_forecasts: dict[str, list[float]],
    lstm_forecasts: dict[str, list[float]],
) -> None:
    fig, axes = plt.subplots(len(tickers), 1, figsize=(14, 12), sharex=False)
    if len(tickers) == 1:
        axes = [axes]

    for i, ticker in enumerate(tickers):
        ax = axes[i]
        hist = recent_actuals[ticker]
        f_rnn = np.asarray(rnn_forecasts[ticker], dtype=np.float32)
        f_lstm = np.asarray(lstm_forecasts[ticker], dtype=np.float32)
        n_hist = len(hist)
        x_hist = np.arange(n_hist)
        x_fore = np.arange(n_hist, n_hist + len(f_rnn))

        ax.plot(x_hist, hist, color="black", label="Last 60 Actual")
        ax.plot(x_fore, f_rnn, color="tab:blue", linestyle="--", label="RNN 30-day Forecast")
        ax.plot(x_fore, f_lstm, color="tab:orange", linestyle="--", label="LSTM 30-day Forecast")
        ax.axvline(n_hist - 0.5, color="gray", linestyle="--", linewidth=1)

        ax.annotate(
            f"RNN: {f_rnn[-1]:.2f}",
            xy=(x_fore[-1], f_rnn[-1]),
            xytext=(5, 5),
            textcoords="offset points",
            color="tab:blue",
            fontsize=9,
        )
        ax.annotate(
            f"LSTM: {f_lstm[-1]:.2f}",
            xy=(x_fore[-1], f_lstm[-1]),
            xytext=(5, -12),
            textcoords="offset points",
            color="tab:orange",
            fontsize=9,
        )

        ax.set_title(f"{ticker} — 30-Day Forecast")
        ax.set_ylabel("USD")
        ax.grid(alpha=0.2)
        if i == 0:
            ax.legend(loc="upper left", ncol=3, fontsize=9)

    axes[-1].set_xlabel("Relative day index")
    plt.tight_layout()
    plt.savefig("plot_03_forecast.png", dpi=150)
    plt.close()


def plot_metrics(tickers: list[str], metrics: dict[str, dict[str, dict[str, float]]]) -> None:
    x = np.arange(len(tickers))
    width = 0.35
    rnn_rmse = [metrics[t]["RNN"]["RMSE"] for t in tickers]
    lstm_rmse = [metrics[t]["LSTM"]["RMSE"] for t in tickers]
    rnn_mae = [metrics[t]["RNN"]["MAE"] for t in tickers]
    lstm_mae = [metrics[t]["LSTM"]["MAE"] for t in tickers]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    bars1 = ax1.bar(x - width / 2, rnn_rmse, width=width, color="tab:blue", alpha=0.85, label="RNN RMSE")
    bars2 = ax1.bar(
        x + width / 2, lstm_rmse, width=width, color="tab:orange", alpha=0.85, label="LSTM RMSE"
    )
    ax1.set_ylabel("RMSE (USD)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tickers)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    line1, = ax2.plot(x, rnn_mae, "o--", color="tab:blue", label="RNN MAE")
    line2, = ax2.plot(x, lstm_mae, "s--", color="tab:orange", label="LSTM MAE")
    ax2.set_ylabel("MAE (USD)")

    handles = [bars1, bars2, line1, line2]
    labels = ["RNN RMSE", "LSTM RMSE", "RNN MAE", "LSTM MAE"]
    ax1.legend(handles, labels, loc="upper left", ncol=2)
    plt.title("RMSE Bars + MAE Lines by Ticker")
    plt.tight_layout()
    plt.savefig("plot_04_metrics.png", dpi=150)
    plt.close()


def plot_residuals(
    tickers: list[str],
    actuals: dict[str, np.ndarray],
    rnn_preds: dict[str, np.ndarray],
    lstm_preds: dict[str, np.ndarray],
) -> None:
    from scipy.stats import gaussian_kde  # local import per requirement

    fig, axes = plt.subplots(2, len(tickers), figsize=(15, 8), sharex=False, sharey=False)
    if len(tickers) == 1:
        axes = np.array([[axes[0]], [axes[1]]], dtype=object)

    model_names = ["RNN", "LSTM"]
    colors = {"RNN": "steelblue", "LSTM": "darkorange"}

    for c, ticker in enumerate(tickers):
        residuals = {
            "RNN": actuals[ticker] - rnn_preds[ticker],
            "LSTM": actuals[ticker] - lstm_preds[ticker],
        }

        for r, model_name in enumerate(model_names):
            ax = axes[r, c]
            vals = residuals[model_name]
            sns.histplot(vals, bins=30, kde=False, color=colors[model_name], alpha=0.75, ax=ax)
            try:
                kde = gaussian_kde(vals)
                x_grid = np.linspace(vals.min(), vals.max(), 200)
                y_grid = kde(x_grid)
                scale = len(vals) * (vals.max() - vals.min()) / 30.0 if vals.max() > vals.min() else 1.0
                ax.plot(x_grid, y_grid * scale, color="black", linewidth=1.2)
            except Exception:
                pass

            ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            ax.text(
                0.03,
                0.95,
                f"mean={mean_val:.2f}\nstd={std_val:.2f}",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
            )
            ax.set_title(f"{model_name} — {ticker}")
            if c == 0:
                ax.set_ylabel("Count")
            ax.set_xlabel("Residual (actual - predicted)")

    plt.tight_layout()
    plt.savefig("plot_05_residuals.png", dpi=150)
    plt.close()


def plot_sensitivity_heatmap(
    tickers: list[str],
    rnn_sensitivity: dict[str, np.ndarray],
    lstm_sensitivity: dict[str, np.ndarray],
) -> None:
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for ticker in tickers:
        rows.append(rnn_sensitivity[ticker])
        labels.append(f"RNN-{ticker}")
        rows.append(lstm_sensitivity[ticker])
        labels.append(f"LSTM-{ticker}")

    mat = np.vstack(rows)
    plt.figure(figsize=(14, 6))
    im = plt.imshow(mat, aspect="auto", cmap="YlOrRd")
    plt.colorbar(im, label="Normalised |gradient|")
    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(0, SEQ_LEN, 5))
    plt.xlabel("Timestep (0=oldest, 59=most recent)")
    plt.ylabel("Model-Ticker")
    plt.title(
        "Input Sensitivity Heatmap: RNN often peaks near timestep 59 while "
        "LSTM tends to distribute sensitivity more broadly (vanishing gradient effect)"
    )
    plt.tight_layout()
    plt.savefig("plot_06_sensitivity_heatmap.png", dpi=150)
    plt.close()


def plot_portfolio_growth(
    rnn_portfolio: np.ndarray,
    lstm_portfolio: np.ndarray,
    bah_portfolio: np.ndarray,
    rnn_holdings: list[str],
    lstm_holdings: list[str],
) -> None:
    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(3, 1, height_ratios=[6, 1, 1], hspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_bot = fig.add_subplot(gs[2, 0], sharex=ax_top)

    x = np.arange(len(rnn_portfolio))
    ax_top.plot(x, rnn_portfolio, color="tab:blue", label="RNN switching", linewidth=2)
    ax_top.plot(x, lstm_portfolio, color="tab:orange", label="LSTM switching", linewidth=2)
    ax_top.plot(x, bah_portfolio, color="black", linestyle="--", label="Equal-weight B&H", linewidth=1.8)
    ax_top.plot(x, np.full_like(x, INITIAL_CAPITAL, dtype=np.float64), color="gray", linestyle=":", label="Cash")
    ax_top.fill_between(x, bah_portfolio, color="black", alpha=0.08)
    ax_top.set_ylabel("Portfolio Value (USD)")
    ax_top.grid(alpha=0.25)
    ax_top.legend(loc="upper left", ncol=4, fontsize=9)

    ax_top.annotate(
        f"{rnn_portfolio[-1]:.2f}",
        xy=(x[-1], rnn_portfolio[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color="tab:blue",
        fontsize=9,
    )
    ax_top.annotate(
        f"{lstm_portfolio[-1]:.2f}",
        xy=(x[-1], lstm_portfolio[-1]),
        xytext=(6, -12),
        textcoords="offset points",
        color="tab:orange",
        fontsize=9,
    )
    ax_top.annotate(
        f"{bah_portfolio[-1]:.2f}",
        xy=(x[-1], bah_portfolio[-1]),
        xytext=(6, 10),
        textcoords="offset points",
        color="black",
        fontsize=9,
    )

    mapping = {"Cash": 0, "AAPL": 1, "MSFT": 2, "GOOGL": 3}
    cmap_colors = ["lightgrey", "royalblue", "seagreen", "tomato"]
    strip_rnn = np.array([mapping.get(h, 0) for h in rnn_holdings], dtype=np.int32)[np.newaxis, :]
    strip_lstm = np.array([mapping.get(h, 0) for h in lstm_holdings], dtype=np.int32)[np.newaxis, :]

    ax_mid.imshow(strip_rnn, aspect="auto", cmap=matplotlib.colors.ListedColormap(cmap_colors), vmin=0, vmax=3)
    ax_bot.imshow(strip_lstm, aspect="auto", cmap=matplotlib.colors.ListedColormap(cmap_colors), vmin=0, vmax=3)

    ax_mid.set_yticks([])
    ax_bot.set_yticks([])
    ax_mid.set_ylabel("RNN\nHold", rotation=0, labelpad=25, va="center")
    ax_bot.set_ylabel("LSTM\nHold", rotation=0, labelpad=25, va="center")
    ax_bot.set_xlabel("Trading day index")
    ax_mid.tick_params(axis="x", labelbottom=False)

    legend_elems = [
        Patch(facecolor="royalblue", label="AAPL"),
        Patch(facecolor="seagreen", label="MSFT"),
        Patch(facecolor="tomato", label="GOOGL"),
        Patch(facecolor="lightgrey", label="Cash"),
    ]
    ax_bot.legend(handles=legend_elems, loc="upper left", ncol=4, fontsize=8, framealpha=0.9)

    plt.tight_layout()
    plt.savefig("plot_07_portfolio_growth.png", dpi=150)
    plt.close()


# =========================
# 9) Terminal Output
# =========================
def summarize_strategy(values: np.ndarray) -> dict[str, float]:
    daily_ret = np.diff(values) / np.clip(values[:-1], 1e-8, None)
    if len(daily_ret) == 0:
        best_day = 0.0
        worst_day = 0.0
    else:
        best_day = float(np.max(daily_ret) * 100)
        worst_day = float(np.min(daily_ret) * 100)
    final_val = float(values[-1])
    total_ret = (final_val / INITIAL_CAPITAL - 1.0) * 100
    return {"final": final_val, "return_pct": total_ret, "best_day_pct": best_day, "worst_day_pct": worst_day}


def top_holdings(holdings: list[str], k: int = 3) -> list[tuple[str, int]]:
    counts = Counter(holdings)
    return counts.most_common(k)


def print_summary(
    rnn_portfolio: np.ndarray,
    lstm_portfolio: np.ndarray,
    bah_portfolio: np.ndarray,
    rnn_switches: int,
    lstm_switches: int,
    rnn_holdings: list[str],
    lstm_holdings: list[str],
) -> None:
    rnn_stats = summarize_strategy(rnn_portfolio)
    lstm_stats = summarize_strategy(lstm_portfolio)
    bah_stats = summarize_strategy(bah_portfolio)

    print("\n| Strategy          | Start  | Final    | Return  | Best day | Worst day | # Switches |")
    print("|-------------------|--------|----------|---------|----------|-----------|------------|")
    print(
        f"| RNN switching     | $1000  | ${rnn_stats['final']:,.2f} | "
        f"{rnn_stats['return_pct']:+.1f}% | {rnn_stats['best_day_pct']:+.1f}%    | "
        f"{rnn_stats['worst_day_pct']:+.1f}%     | {rnn_switches:>2d}         |"
    )
    print(
        f"| LSTM switching    | $1000  | ${lstm_stats['final']:,.2f} | "
        f"{lstm_stats['return_pct']:+.1f}% | {lstm_stats['best_day_pct']:+.1f}%    | "
        f"{lstm_stats['worst_day_pct']:+.1f}%     | {lstm_switches:>2d}         |"
    )
    print(
        f"| Equal-weight B&H  | $1000  | ${bah_stats['final']:,.2f} | "
        f"{bah_stats['return_pct']:+.1f}% | {bah_stats['best_day_pct']:+.1f}%    | "
        f"{bah_stats['worst_day_pct']:+.1f}%     |  0         |"
    )

    print("\nTop 3 most-held positions (RNN):")
    for name, days in top_holdings(rnn_holdings):
        print(f"  - {name}: {days} days")

    print("Top 3 most-held positions (LSTM):")
    for name, days in top_holdings(lstm_holdings):
        print(f"  - {name}: {days} days")


# =========================
# 10) Main Orchestration
# =========================
def make_loaders(
    train_arr: np.ndarray, test_arr: np.ndarray, seq_len: int, batch_size: int
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
]:
    val_fraction = 0.2
    split = int(len(train_arr) * (1 - val_fraction))
    split = max(seq_len + 1, split)
    train_part = train_arr[:split]
    val_part = train_arr[split - seq_len :]

    train_ds = StockDataset(train_part, seq_len)
    val_ds = StockDataset(val_part, seq_len)
    test_ds = StockDataset(test_arr, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main() -> None:
    print(f"Using device: {DEVICE}")
    data = download_data()
    if not data:
        raise RuntimeError("No ticker data was downloaded. Exiting.")

    valid_tickers = [t for t in TICKERS if t in data]
    if len(valid_tickers) == 0:
        raise RuntimeError("No valid tickers available after download.")

    all_actuals: dict[str, np.ndarray] = {}
    all_rnn_preds: dict[str, np.ndarray] = {}
    all_lstm_preds: dict[str, np.ndarray] = {}
    all_metrics: dict[str, dict[str, dict[str, float]]] = {}
    all_recent_actuals: dict[str, np.ndarray] = {}
    all_rnn_forecasts: dict[str, list[float]] = {}
    all_lstm_forecasts: dict[str, list[float]] = {}
    all_rnn_sens: dict[str, np.ndarray] = {}
    all_lstm_sens: dict[str, np.ndarray] = {}

    for ticker in valid_tickers:
        print(f"\n=== Processing {ticker} ===")
        train_arr, test_arr, price_scaler, _ = prepare_data(data[ticker], SEQ_LEN, test_size=0.2)
        train_loader, val_loader, test_loader = make_loaders(train_arr, test_arr, SEQ_LEN, BATCH_SIZE)

        rnn = RNNModel(
            input_size=train_arr.shape[1],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)
        lstm = LSTMModel(
            input_size=train_arr.shape[1],
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
        ).to(DEVICE)

        rnn_train, rnn_val, lstm_train, lstm_val = train_model(
            rnn, lstm, train_loader, val_loader, EPOCHS, LR, DEVICE
        )
        plot_loss_curves(ticker, rnn_train, rnn_val, lstm_train, lstm_val)

        rnn_preds, actuals, rnn_metrics = evaluate_model(rnn, test_loader, price_scaler, DEVICE)
        lstm_preds, actuals_lstm, lstm_metrics = evaluate_model(lstm, test_loader, price_scaler, DEVICE)

        if len(actuals) != len(actuals_lstm):
            min_len = min(len(actuals), len(actuals_lstm))
            actuals = actuals[:min_len]
            rnn_preds = rnn_preds[:min_len]
            lstm_preds = lstm_preds[:min_len]
        else:
            actuals = actuals

        all_actuals[ticker] = actuals
        all_rnn_preds[ticker] = rnn_preds
        all_lstm_preds[ticker] = lstm_preds
        all_metrics[ticker] = {"RNN": rnn_metrics, "LSTM": lstm_metrics}

        last_seq = test_arr[-SEQ_LEN:]
        all_rnn_forecasts[ticker] = predict_future(rnn, last_seq, price_scaler, steps=30, device=DEVICE)
        all_lstm_forecasts[ticker] = predict_future(lstm, last_seq, price_scaler, steps=30, device=DEVICE)

        recent_scaled = test_arr[-SEQ_LEN:, 0]
        recent_usd = inverse_target_from_price_scaler(recent_scaled, price_scaler)
        all_recent_actuals[ticker] = recent_usd

        all_rnn_sens[ticker] = compute_sensitivity(rnn, last_seq, DEVICE)
        all_lstm_sens[ticker] = compute_sensitivity(lstm, last_seq, DEVICE)

    plot_test_predictions(valid_tickers, all_actuals, all_rnn_preds, all_lstm_preds, all_metrics)
    plot_forecast(valid_tickers, all_recent_actuals, all_rnn_forecasts, all_lstm_forecasts)
    plot_metrics(valid_tickers, all_metrics)
    plot_residuals(valid_tickers, all_actuals, all_rnn_preds, all_lstm_preds)
    plot_sensitivity_heatmap(valid_tickers, all_rnn_sens, all_lstm_sens)

    aligned_len = min(len(all_actuals[t]) for t in valid_tickers)
    aligned_rnn_preds = {t: all_rnn_preds[t][:aligned_len] for t in valid_tickers}
    aligned_lstm_preds = {t: all_lstm_preds[t][:aligned_len] for t in valid_tickers}
    aligned_actuals = {t: all_actuals[t][:aligned_len] for t in valid_tickers}

    n_days = aligned_len - 1
    if n_days < 1:
        raise RuntimeError("Not enough aligned test days to run portfolio simulation.")

    rnn_portfolio, rnn_holdings, rnn_switches = simulate_portfolio(
        aligned_rnn_preds, aligned_actuals, n_days=n_days, initial_capital=INITIAL_CAPITAL
    )
    lstm_portfolio, lstm_holdings, lstm_switches = simulate_portfolio(
        aligned_lstm_preds, aligned_actuals, n_days=n_days, initial_capital=INITIAL_CAPITAL
    )
    bah_portfolio = compute_bah_portfolio(aligned_actuals, n_days=n_days, initial_capital=INITIAL_CAPITAL)

    plot_portfolio_growth(rnn_portfolio, lstm_portfolio, bah_portfolio, rnn_holdings, lstm_holdings)

    print_summary(
        rnn_portfolio,
        lstm_portfolio,
        bah_portfolio,
        rnn_switches,
        lstm_switches,
        rnn_holdings,
        lstm_holdings,
    )
    print("\nSaved plots:")
    print(" - plot_01_loss_curves_{ticker}.png")
    print(" - plot_02_test_predictions.png")
    print(" - plot_03_forecast.png")
    print(" - plot_04_metrics.png")
    print(" - plot_05_residuals.png")
    print(" - plot_06_sensitivity_heatmap.png")
    print(" - plot_07_portfolio_growth.png")


if __name__ == "__main__":
    main()
