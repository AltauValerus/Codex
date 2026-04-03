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


## Defaults chosen for your use case

Based on your answers, I picked these defaults:
- Exchange/pair: **Coinbase BTC/USD** (closest to USD reference style).
- Compute: **Colab Free**.
- Objective: **direction with cost-aware decision filter** (matches up/down market usage).
- Decision cost fallback: **0.02** (2%) until you provide a better estimate.
- Confidence edge threshold: **0.03** to avoid low-conviction bets.

Example baseline run:

```bash
python src/train/train_baseline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --decision-cost 0.02 \
  --min-edge 0.03 \
  --out-dir artifacts/baseline
```

Config preset: `configs/polymarket_colab_free.json`.


## Next step after baseline: train an image CNN

```bash
python src/train/train_image_cnn.py \
  --images-dir data/processed/images_224 \
  --windows data/interim/windows/window_index.parquet \
  --epochs 8 \
  --batch-size 64 \
  --lr 1e-3 \
  --out-dir artifacts/image_cnn
```

Outputs:
- `artifacts/image_cnn/image_cnn.pt`
- `artifacts/image_cnn/metrics.json`

Use this only after the baseline is running; compare out-of-sample metrics against the baseline before trusting it.


## Offline smoke-test mode (no exchange network)

If exchange APIs are unreachable, generate synthetic OHLCV and run the full pipeline offline:

```bash
python src/data/generate_synthetic_ohlcv.py \
  --start 2026-01-01T00:00:00Z \
  --periods 2500 \
  --timeframe-minutes 5 \
  --out data/raw/ohlcv/synthetic_btcusd_5m.parquet
```

Then use `synthetic_btcusd_5m.parquet` in the same build/train commands.


## Phase-1 tools added (labeling + walk-forward eval)

Build advanced labels:

```bash
python src/data/build_labels.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --method fixed_horizon_ternary \
  --horizon 1 \
  --flat-threshold 0.0005 \
  --out data/interim/labels/fixed_horizon_ternary.parquet

python src/data/build_labels.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --method triple_barrier \
  --horizon 12 \
  --pt 0.0025 \
  --sl 0.0025 \
  --out data/interim/labels/triple_barrier.parquet
```

Walk-forward evaluation:

```bash
python src/eval/walk_forward.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-bars 400 \
  --test-bars 100 \
  --step-bars 100 \
  --out artifacts/eval/walk_forward_report.json
```


## Next baseline upgrade: gradient boosting

```bash
python src/train/train_gbdt_baseline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --decision-cost 0.02 \
  --min-edge 0.03 \
  --out-dir artifacts/gbdt_baseline
```

Outputs:
- `artifacts/gbdt_baseline/gbdt_direction_model.joblib`
- `artifacts/gbdt_baseline/metrics.json`


## Next model upgrade: sequence GRU baseline

```bash
python src/train/train_sequence_model.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --lookback 48 \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-3 \
  --out-dir artifacts/sequence_gru
```

Outputs:
- `artifacts/sequence_gru/gru_sequence_model.pt`
- `artifacts/sequence_gru/metrics.json`


## Compare models in one report (ablation helper)

```bash
python src/eval/compare_models.py \
  --model logistic=artifacts/baseline/metrics.json \
  --model gbdt=artifacts/gbdt_baseline/metrics.json \
  --model gru=artifacts/sequence_gru/metrics.json \
  --model cnn=artifacts/image_cnn/metrics.json \
  --out artifacts/eval/model_comparison.json \
  --out-csv artifacts/eval/model_comparison.csv
```

This produces a single comparison table and reports missing model files without crashing.


## One-command ablation pipeline (judgment-call helper)

If you want one command to train key models and auto-compare them:

```bash
python src/eval/run_ablation_pipeline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --images-dir data/processed/images_224 \
  --windows data/interim/windows/window_index.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --out-root artifacts/ablation
```

It trains logistic, GBDT, and GRU.
If images/windows exist, it also runs CNN.
Then it creates:
- `artifacts/ablation/model_comparison.json`
- `artifacts/ablation/model_comparison.csv`
