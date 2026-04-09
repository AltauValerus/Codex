# Operator Runbook

This runbook is the canonical execution guide for day-to-day use.

## Mode 1: Single-model (default)

Use this for routine operation when you already selected a model family and want a fast decision loop.

```bash
python -m src.eval.run_single_model_pipeline \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --out-root artifacts/single_model
```

Outputs:
- `artifacts/single_model/baseline/metrics.json`
- `artifacts/single_model/backtest_baseline.json`
- `artifacts/single_model/readiness_baseline.json`
- `artifacts/single_model/go_no_go_baseline.json`

## Mode 2: Ablation (periodic benchmarking)

Use this when you want to compare multiple model families and re-select the best candidate.

```bash
python -m src.eval.run_ablation_pipeline \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --images-dir data/processed/images_224 \
  --windows data/interim/windows/window_index.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --gate-model best \
  --out-root artifacts/ablation
```

Notes:
- `--gate-model best` attempts to use the best model by test AUC for backtest/go-no-go.
- Current backtest support is implemented for `logistic` and `gbdt`.
- Unsupported gate models fall back to `logistic` with a warning.

## Operational policy

1. Run **single-model** mode by default.
2. Run **ablation** weekly/biweekly or after major data/feature changes.
3. Update thresholds (`min_*`) only with documented rationale.
