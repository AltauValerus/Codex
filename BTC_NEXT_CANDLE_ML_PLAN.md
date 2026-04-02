# BTC 5-Minute Next-Candle Prediction Plan (Image + Time-Series)

## 1) Objective and framing

**Goal:** Predict the *next 5-minute candle* for BTC from recent candle history.

Use two target formulations in parallel:
1. **Direction classification**: next close > current close (UP/DOWN).
2. **Magnitude regression**: next return \(r_{t+1} = \log(C_{t+1}/C_t)\).

Track both, but optimize for **trading-aware metrics** and calibration, not just raw accuracy.

---

## 2) Data strategy ("go back in time")

For best ML standards, build a **reproducible historical dataset** rather than scraping random screenshots.

### 2.1 Primary market data (ground truth)
Collect historical 5-minute OHLCV from reliable exchange APIs (Binance, Coinbase, Kraken) for the same periods:
- `timestamp_open`, `open`, `high`, `low`, `close`, `volume`
- optional: order book imbalance, funding/open interest (if available)

### 2.2 Generate chart images from historical windows (preferred over manual photos)
Instead of literal photos, render deterministic chart images from OHLC windows:
- Sliding window length: **96 candles** (8 hours) or **144 candles** (12 hours)
- Render a candlestick chart image at time `t` from candles `[t-L+1, ..., t]`
- Label is candle at `t+1`

This avoids screenshot drift and keeps train/val/test reproducible.

### 2.3 If you still want real screenshots from the site
Automate with Playwright/Selenium:
- load historical replay view (if supported)
- fixed viewport, dark/light theme fixed, no overlays
- capture PNG every 5 minutes equivalent in replay
- store metadata JSON with exact timestamp and symbol

---

## 3) Dataset specification

### 3.1 Folder layout

```text
/data
  /raw
    /ohlcv/binance_btcusdt_5m.parquet
  /interim
    /windows/window_index.parquet
  /processed
    /images_224/
      train/*.png
      val/*.png
      test/*.png
    labels.parquet
```

### 3.2 Labeling
For each sample ending at candle `t`:
- `y_dir = 1 if close[t+1] > close[t] else 0`
- `y_ret = log(close[t+1]/close[t])`
- optional confidence class: `|y_ret|` buckets

### 3.3 Strict time splits (no leakage)
Use chronological split, e.g.:
- Train: 2021-01-01 → 2024-06-30
- Validation: 2024-07-01 → 2025-03-31
- Test: 2025-04-01 → 2026-03-31

Never shuffle across time.

---

## 4) Feature and modeling approach (best-practice)

## 4.1 Baselines first (must beat)
1. **Naive persistence**: next direction = current direction
2. **Linear/logistic models** on handcrafted indicators
3. **Gradient boosting (XGBoost/LightGBM)** on numerical features

If deep model cannot outperform robust baselines on out-of-sample + costs, stop.

### 4.2 Image model path
- Backbone: EfficientNetV2-S / ConvNeXt-Tiny
- Input: 224x224 or 256x256 candlestick image
- Loss:
  - classification: weighted BCE / focal loss
  - regression: Huber loss
- Regularization: label smoothing, dropout, weight decay

### 4.3 Stronger hybrid model (recommended)
Use two-branch architecture:
- CNN branch for chart image
- Temporal branch (TCN/LSTM/Transformer) for raw OHLCV sequence
- Fuse representations -> multi-head output (direction + return)

This usually outperforms image-only models because raw prices preserve detail.

---

## 5) Training protocol (Colab-friendly)

### 5.1 Reproducibility
- Set global seeds
- Save exact package versions (`requirements.txt`)
- Version dataset snapshots (DVC or immutable parquet files)
- Log everything in W&B/MLflow

### 5.2 Colab workflow
1. Mount Drive / connect HF dataset bucket
2. Load frozen train/val/test split manifests
3. Train with mixed precision + early stopping
4. Save best checkpoint + preprocessing config
5. Export `onnx` / `torchscript` for inference

### 5.3 Hyperparameter tuning
- Optuna/Bayes search on:
  - window length L
  - lr, batch size, weight decay
  - focal gamma, class weights
  - threshold for decision rule

Do tuning on validation only; test set touched once at the end.

---

## 6) Evaluation standards

### 6.1 ML metrics
- Classification: AUC, F1, balanced accuracy, Brier score, ECE calibration
- Regression: MAE, RMSE, sign accuracy, quantile loss

### 6.2 Trading-aware evaluation
Backtest with realistic assumptions:
- fees + slippage
- latency (decision executed next bar open)
- turnover constraints

Report:
- Sharpe/Sortino
- max drawdown
- hit rate and expectancy
- profit factor

### 6.3 Robustness checks
- Walk-forward validation (rolling retrain)
- Regime split (high/low volatility)
- Bootstrap CIs for key metrics
- Ablation: image-only vs sequence-only vs hybrid

---

## 7) Deployment plan

### 7.1 Inference service
- Scheduled every 5 minutes
- Pull latest candles, build features/image, run model
- Output probability + confidence + suggested action (if any)

### 7.2 Monitoring
- Prediction drift and feature drift
- Calibration drift
- Live vs backtest delta
- Auto-retrain trigger rules

### 7.3 Safe rollout
- Paper-trading first (2–4 weeks)
- Small capital canary phase
- Hard risk limits (max positions, max daily loss, kill switch)

---

## 8) Concrete data collection plan for “photos”

1. Download full historical 5m OHLCV for BTC.
2. Build a window index of all valid timestamps.
3. For each timestamp, render a standardized candlestick image (not manual screenshot).
4. Save image + label in immutable train/val/test manifests.
5. Add data quality checks:
   - missing bars
   - duplicated timestamps
   - timezone consistency UTC
6. Freeze dataset v1.0 and publish to Hugging Face Dataset repo.

---

## 9) Suggested tool stack

- **Data**: pandas/polars, pyarrow, ccxt, exchange native APIs
- **Modeling**: PyTorch + timm + Lightning
- **Tuning**: Optuna
- **Tracking**: W&B or MLflow
- **Versioning**: DVC + Git LFS (or HF dataset artifacts)
- **Hosting**: Hugging Face Inference Endpoint / FastAPI on a VM
- **Automation**: GitHub Actions + cron/worker

---

## 10) Important caution

Short-horizon BTC next-candle prediction is noisy and edge is often tiny after fees/slippage.
Treat this as a research pipeline first:
- prioritize leakage control,
- robustness,
- and realistic execution assumptions.

If you want, the next step is a **day-by-day implementation checklist** (data script, renderer, training notebook, backtest notebook, deployment skeleton) formatted for Colab + GitHub + Hugging Face.
