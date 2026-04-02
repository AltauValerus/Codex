# BTC Next-Candle Dataset Starter

This repository now includes a concrete starter pipeline for the **next step** after planning:
1. fetch historical BTC 5m OHLCV,
2. build leakage-safe window labels,
3. render deterministic candlestick images for train/val/test.

## Quickstart

```bash
pip install -r requirements.txt

python src/data/fetch_ohlcv.py \
  --exchange binance \
  --symbol BTC/USDT \
  --timeframe 5m \
  --start 2021-01-01T00:00:00Z \
  --end 2026-03-31T23:55:00Z \
  --out data/raw/ohlcv/binance_btcusdt_5m.parquet

python src/data/build_windows.py \
  --ohlcv data/raw/ohlcv/binance_btcusdt_5m.parquet \
  --window-size 96 \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --out data/interim/windows/window_index.parquet

python src/data/render_charts.py \
  --ohlcv data/raw/ohlcv/binance_btcusdt_5m.parquet \
  --windows data/interim/windows/window_index.parquet \
  --out-dir data/processed/images_224 \
  --width 224 \
  --height 224
```

## Notes
- All splits are time-based, not random.
- Labels target the **next candle** (`t+1`) from a window ending at `t`.
- For Colab, place data on Drive or object storage and run the same scripts.

See `BTC_NEXT_CANDLE_ML_PLAN.md` for the full methodology.


## Next step (train a baseline now)

After generating data, train a calibrated direction baseline:

```bash
python src/train/train_baseline.py \
  --ohlcv data/raw/ohlcv/binance_btcusdt_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --out-dir artifacts/baseline
```

This produces:
- `artifacts/baseline/baseline_direction_model.joblib`
- `artifacts/baseline/metrics.json`

Use these metrics as the minimum benchmark before training CNN/hybrid models.


## Validate data before training

```bash
python src/data/validate_dataset.py \
  --ohlcv data/raw/ohlcv/binance_btcusdt_5m.parquet \
  --windows data/interim/windows/window_index.parquet \
  --timeframe-minutes 5
```

This checks duplicates/missing gaps, split ordering, label alignment, and class balance.

## What I need from you

Please confirm these 5 items so we can run full training exactly as you want:
1. Exchange + pair (default: Binance BTC/USDT).
2. Exact historical range (default in repo: 2021-01-01 to 2026-03-31 UTC).
3. Whether we optimize for **direction accuracy** or **trading PnL after fees**.
4. Fee + slippage assumptions for backtests.
5. Compute target (Colab Free / Pro / local GPU).
