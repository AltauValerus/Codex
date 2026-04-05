# Project Status

## Have models been trained yet?

Not yet in this environment.

So far, the repository includes:
- data collection + preprocessing scripts,
- dataset validation checks,
- baseline and CNN training scripts,
- configuration presets for Colab Free.

## Why no training run has been executed here

This environment has not been provided with the full BTC dataset artifacts
(`data/raw`, `data/interim`, `data/processed`) required for end-to-end model fitting.

## How to run training now (Colab Free)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Build dataset artifacts:

```bash
python src/data/fetch_ohlcv.py \
  --exchange coinbase \
  --symbol BTC/USD \
  --timeframe 5m \
  --start 2021-01-01T00:00:00Z \
  --end 2026-03-31T23:55:00Z \
  --out data/raw/ohlcv/coinbase_btcusd_5m.parquet

python src/data/build_windows.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --window-size 96 \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --out data/interim/windows/window_index.parquet

python src/data/render_charts.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --windows data/interim/windows/window_index.parquet \
  --out-dir data/processed/images_224 \
  --width 224 \
  --height 224

python src/data/validate_dataset.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --windows data/interim/windows/window_index.parquet \
  --timeframe-minutes 5
```

3. Train baseline:

```bash
python src/train/train_baseline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --decision-cost 0.02 \
  --min-edge 0.03 \
  --out-dir artifacts/baseline
```

4. Train image CNN:

```bash
python src/train/train_image_cnn.py \
  --images-dir data/processed/images_224 \
  --windows data/interim/windows/window_index.parquet \
  --epochs 8 \
  --batch-size 64 \
  --lr 1e-3 \
  --out-dir artifacts/image_cnn
```

## Expected artifacts

- `artifacts/baseline/baseline_direction_model.joblib`
- `artifacts/baseline/metrics.json`
- `artifacts/image_cnn/image_cnn.pt`
- `artifacts/image_cnn/metrics.json`
