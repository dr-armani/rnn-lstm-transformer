# RNN vs LSTM Stock Forecasting — Cursor Prompt

**Role:** Act as a Senior AI Research Engineer and Educator.

**Objective:** Write a complete, self-contained, production-quality Python script named
`rnn_vs_lstm_demo.py` that compares a Vanilla RNN and an LSTM for multi-stock price
forecasting. Every function must be fully implemented — no placeholder comments, no
stubs. The script must be runnable immediately after setup with a single command.

---

## 0. Environment Setup

Include these instructions as a comment block at the top of the script:

```
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
```

---

## 1. Imports & Global Configuration

```python
TICKERS         = ['AAPL', 'MSFT', 'GOOGL']
SEQ_LEN         = 60       # lookback window (trading days)
BATCH_SIZE      = 32
EPOCHS          = 150
LR              = 0.001
HIDDEN_SIZE     = 64
NUM_LAYERS      = 2
DROPOUT         = 0.2
INITIAL_CAPITAL = 1000.0
SEED            = 1
```

Set `matplotlib.use('Agg')` immediately after importing matplotlib so plots render
correctly in all environments (no display required).

Device detection — support all three backends in priority order:
```python
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.cuda.manual_seed_all(SEED)
elif torch.backends.mps.is_available():   # Apple Silicon
    device = torch.device('mps')
else:
    device = torch.device('cpu')
```
Also seed `torch`, `numpy`, and `random` to `SEED`.

---

## 2. Data Engineering

**`download_data() -> dict[str, pd.DataFrame]`**
- Download ~4 years of OHLCV data for each ticker via
  `yfinance.download(ticker, auto_adjust=False, progress=False)`
- Wrap each download in a try/except; skip and warn if a ticker fails rather than
  crashing the whole script
- Return `{ticker: DataFrame}` with a `DatetimeIndex`

**`prepare_data(df, seq_len, test_size=0.2) -> (train, test, price_scaler, vol_scaler)`**
- Features: `['Adj Close', 'High', 'Low']` + `['Volume']`; fall back to `Close` if
  `Adj Close` is absent
- Scale prices and Volume with separate `MinMaxScaler(0,1)` instances — Volume's
  magnitude (millions of shares) would dominate price features if scaled together
- Build the scaled array as `np.hstack([price_scaled, vol_scaled])` → shape `(N, 4)`
- 80/20 temporal split — no shuffling; the test slice must include `seq_len` rows of
  context so the first test window is valid: `test = data[split - seq_len:]`
- Column 0 of the scaled array is always Adj Close (the prediction target)

**`StockDataset(Dataset)`**
- `__getitem__`: returns `(x, y)` where `x = data[i:i+seq_len]` and
  `y = data[i+seq_len, 0]` (next-day Adj Close, scaled)
- Train DataLoader: `shuffle=True`; test/val DataLoader: `shuffle=False`

---

## 3. Model Architectures

Two structurally parallel PyTorch classes — only the recurrent cell differs:

**`RNNModel`**
```
nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
nn.Linear(hidden_size, 1)
```
Forward: initialise `h0` only → pass through RNN → apply linear head to `out[:, -1, :]`

**`LSTMModel`**
```
nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
nn.Linear(hidden_size, 1)
```
Forward: initialise both `h0` and `c0` → pass through LSTM → apply linear head to
`out[:, -1, :]`

The cell state `c0` is the long-term memory that separates LSTM from RNN — keep this
difference visually obvious in the code.

---

## 4. Training

**`train_model(rnn, lstm, train_loader, val_loader, epochs, lr) -> (rnn_tl, rnn_vl, lstm_tl, lstm_vl)`**

Train both models in the same loop so they see identical batches:
1. Train step: forward → MSELoss → backward → Adam step (separately for each model)
2. Val step: `model.eval()` + `torch.no_grad()` over val_loader at end of each epoch
3. Record four loss lists: `rnn_train`, `rnn_val`, `lstm_train`, `lstm_val`
4. Print every 10 epochs:
   `Epoch [ X/150]  RNN  — Train: X.XXXX | Val: X.XXXX   LSTM — Train: X.XXXX | Val: X.XXXX`

---

## 5. Evaluation & Forecasting

**`evaluate_model(model, test_loader, price_scaler) -> (preds, actuals, metrics)`**
- Collect scaled predictions and actuals from the test DataLoader
- Inverse-transform using the dummy-array approach (only price_scaler is needed):
  create a zeros array of shape `(N, price_scaler.n_features_in_)`, fill column 0
  with scaled values, call `inverse_transform`, extract column 0
- Compute and return: RMSE, MAE, MAPE (on USD values)

**`predict_future(model, last_sequence, price_scaler, steps=30) -> list[float]`**
- Closed-loop autoregressive: predict one step → append to window → drop oldest step
- When rolling the window forward for multi-feature input, hold non-target features
  fixed at their last known values
- Inverse-transform each prediction before appending to the output list

---

## 6. Input Sensitivity

**`compute_sensitivity(model, last_sequence) -> np.ndarray`** (shape: `seq_len`)
- Set `x.requires_grad_(True)`, run forward pass, call `.backward()` on the scalar
  output
- Take `x.grad`, average absolute value over the feature dimension → shape `(seq_len,)`
- Normalise to [0, 1] by dividing by max
- This reveals which timesteps in the 60-day window each model attends to. Vanilla RNNs
  peak near timestep 59 (most recent) due to vanishing gradients; LSTMs show broader
  sensitivity. This is the vanishing gradient effect made visual.

---

## 7. Portfolio Simulation

**`simulate_portfolio(all_preds, all_actuals, n_days) -> (portfolio, holdings, switches)`**

Multi-stock switching strategy — run separately for RNN signals and LSTM signals:
- Starting capital: `$1,000`
- Each day: compute predicted return per ticker = `(pred - today_actual) / today_actual`
- Invest 100% of capital in the ticker with the highest predicted positive return;
  if no ticker is predicted to rise, hold Cash
- Realise the gain/loss at next day's actual close price
- Track daily portfolio value, which stock was held (or "Cash"), and switch count

**`compute_bah_portfolio(all_actuals, n_days) -> np.ndarray`**
- Equal-weight buy-and-hold: `$333.33` in each ticker on day 1, hold to end
- Align test lengths across tickers by taking the minimum length before running either
  strategy

---

## 8. Plots

Use `matplotlib.use('Agg')` and save every plot with `plt.savefig(filename, dpi=150)`.
Never call `plt.show()`. Close each figure with `plt.close()` immediately after saving.

**Plot 1 — Loss curves** (`plot_01_loss_curves_{ticker}.png`, one per ticker)
- 4 lines: RNN train (blue solid), RNN val (blue dashed), LSTM train (orange solid),
  LSTM val (orange dashed)
- Log-scale y-axis — overfitting is visible where val diverges from train
- Title: `"Training vs Validation Loss — {ticker}"`

**Plot 2 — Test predictions** (`plot_02_test_predictions.png`)
- 3-row subplot (one per ticker)
- Each row: actual (black), RNN (blue α=0.7), LSTM (orange α=0.7)
- `fill_between` the min/max of RNN and LSTM predictions with purple α=0.15 to
  highlight disagreement zones
- Subtitle each row with both RMSE values

**Plot 3 — 30-day forecast** (`plot_03_forecast.png`)
- 3-row subplot; each row: last 60 actual days (black) + RNN dashed (blue) +
  LSTM dashed (orange)
- Grey vertical dashed line at forecast start
- Annotate final predicted price for each model at the right edge of the axis

**Plot 4 — Metrics bar chart** (`plot_04_metrics.png`)
- Grouped bars by ticker: RNN RMSE (blue) and LSTM RMSE (orange) on primary y-axis
- RNN MAE (blue dashed line + circle markers) and LSTM MAE (orange dashed + square
  markers) on a secondary y-axis
- Four-entry legend

**Plot 5 — Residual histograms** (`plot_05_residuals.png`)
- 2×3 subplot grid (rows = models, columns = tickers)
- Each subplot: histogram of `(actual − predicted)` with KDE overlay (use
  `scipy.stats.gaussian_kde`; wrap in try/except in case scipy is absent), vertical
  red dashed line at zero, text box with mean and std
- Colour: steelblue for RNN rows, darkorange for LSTM rows

**Plot 6 — Input sensitivity heatmap** (`plot_06_sensitivity_heatmap.png`)
- Rows: one per model/ticker combination (6 rows total)
- Columns: timestep 0 (oldest) → 59 (most recent)
- Colormap: `YlOrRd`; add a colorbar labelled "Normalised |gradient|"
- Title must explain what to look for: RNN sensitivity peaks near timestep 59;
  LSTM sensitivity is distributed more broadly — this is the vanishing gradient effect

**Plot 7 — Portfolio growth** (`plot_07_portfolio_growth.png`)
- Single figure with 3 panels (height ratios 6:1:1) sharing the x-axis
- Top panel: RNN strategy (blue), LSTM strategy (orange), equal-weight B&H (black
  dashed), Cash flat line (grey dotted); annotate final USD value for each line;
  light fill under the B&H line for reference
- Middle strip: colour-coded daily holdings for RNN strategy
  (AAPL=royalblue, MSFT=seagreen, GOOGL=tomato, Cash=lightgrey)
- Bottom strip: same for LSTM strategy
- Add a small legend for the colour coding; label each strip on the y-axis

---

## 9. Terminal Output

After all plots are saved, print a formatted summary table:

```
| Strategy          | Start  | Final    | Return  | Best day | Worst day | # Switches |
|-------------------|--------|----------|---------|----------|-----------|------------|
| RNN switching     | $1000  | $X,XXX   | +XX.X%  | +X.X%    | -X.X%     | XX         |
| LSTM switching    | $1000  | $X,XXX   | +XX.X%  | +X.X%    | -X.X%     | XX         |
| Equal-weight B&H  | $1000  | $X,XXX   | +XX.X%  | +X.X%    | -X.X%     | 0          |
```

Also print: top 3 most-held tickers (and days held) for each strategy.

---

## 10. Code Quality

- Single file: `rnn_vs_lstm_demo.py`
- All functions must have Python type hints on every parameter and return value
- Suppress yfinance and sklearn warnings with `warnings.filterwarnings('ignore')`
- Function order: constants → seeds/device → data functions → Dataset → models →
  train → evaluate → forecast → sensitivity → portfolio → plots → summary → `main()`
- `main()` orchestrates the full pipeline: download → prepare → train (per ticker) →
  evaluate → plot → portfolio → summary
- Guard with `if __name__ == '__main__': main()`
