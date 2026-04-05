# Immediate Next Step (Per Plan)

According to `ROADMAP_V2.md`, the immediate next step is **Phase 0**:

## Run the full pipeline on real Coinbase BTC/USD 5m data in Colab, then freeze `dataset_v1`.

Why this is next:
- Current metrics are from synthetic smoke runs only.
- Real-data out-of-sample validation is the first true go/no-go checkpoint.

## Exact command order

1. Fetch real data:

```bash
python src/data/fetch_ohlcv.py \
  --exchange coinbase \
  --symbol BTC/USD \
  --timeframe 5m \
  --start 2021-01-01T00:00:00Z \
  --end 2026-03-31T23:55:00Z \
  --out data/raw/ohlcv/coinbase_btcusd_5m.parquet
```

2. Build windows:

```bash
python src/data/build_windows.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --window-size 96 \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --out data/interim/windows/window_index.parquet
```

3. Validate data quality:

```bash
python src/data/validate_dataset.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --windows data/interim/windows/window_index.parquet \
  --timeframe-minutes 5
```

4. Train strongest numeric baselines first:

```bash
python src/train/train_baseline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --decision-cost 0.02 \
  --min-edge 0.03 \
  --out-dir artifacts/baseline

python src/train/train_gbdt_baseline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --decision-cost 0.02 \
  --min-edge 0.03 \
  --out-dir artifacts/gbdt_baseline
```

5. Run walk-forward check:

```bash
python src/eval/walk_forward.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-bars 400 \
  --test-bars 100 \
  --step-bars 100 \
  --out artifacts/eval/walk_forward_report.json
```

## Pass criteria to move to next phase

- Data validation checks pass.
- At least one numeric baseline shows robust out-of-sample performance after decision costs.
- Walk-forward metrics are stable enough to justify deeper model iterations.
