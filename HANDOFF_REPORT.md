# HANDOFF REPORT — BTC 5m Next-Candle Project

_Last updated: 2026-04-03 (UTC)_

## 1) Project objective

Build a reproducible, leakage-resistant ML pipeline for **BTC/USD 5-minute next-candle prediction** that can:
- classify next-candle direction,
- support cost-aware decision policies,
- and provide clear go/no-go evidence before paper/live usage.

## 2) What has been implemented

### Data pipeline
- `src/data/fetch_ohlcv.py` — exchange OHLCV backfill.
- `src/data/generate_synthetic_ohlcv.py` — synthetic OHLCV for offline smoke tests.
- `src/data/build_windows.py` — sample/window index + next-candle linkage.
- `src/data/build_labels.py` — `fixed_horizon_ternary` + `triple_barrier` labels.
- `src/data/render_charts.py` — deterministic candlestick image rendering.
- `src/data/validate_dataset.py` — duplicates/gaps/split/label checks.

### Training
- `src/train/train_baseline.py` — logistic baseline + calibration + decision metrics.
- `src/train/train_gbdt_baseline.py` — GBDT baseline + calibration + decision metrics.
- `src/train/train_sequence_model.py` — GRU sequence model.
- `src/train/train_image_cnn.py` — image CNN (EfficientNet-b0).

### Evaluation / orchestration
- `src/eval/walk_forward.py` — walk-forward protocol.
- `src/eval/compare_models.py` — aggregate model metrics into single table.
- `src/eval/readiness_report.py` — traffic-light readiness (green/yellow/red).
- `src/eval/backtest_from_model.py` — backtest decision policy from trained model.
- `src/eval/run_ablation_pipeline.py` — one-command multi-model ablation run.

### Docs / planning / status
- Main docs: `README.md`, `BTC_NEXT_CANDLE_ML_PLAN.md`, `ROADMAP_V2.md`.
- Operational docs: `NEXT_ACTION_PLAN.md`, `NEXT_STEP_NOW.md`, `STATUS.md`.
- Analysis docs: `RUN_RESULTS.md`, `EXPERIMENT_SUMMARY.md`, `HONEST_ASSESSMENT.md`, `MODEL_PRIORITY.md`, `WHY_REAL_DATA_NOW.md`, `CURRENT_STATE_BRIEF.md`.

## 3) Current understanding and conclusions

1. **Engineering readiness: good**
   - Pipeline is functional end-to-end.
2. **Research readiness: incomplete**
   - Robust edge is not proven on real data in this environment.
3. **Model priority (current)**
   - Numeric baselines -> sequence -> CNN (only if incremental lift) -> hybrid later.
4. **Not ready for live decisions yet**
   - Need real-data out-of-sample + walk-forward + cost-aware stability first.

## 4) Results obtained so far (environment)

All meaningful runs here are mostly synthetic/offline smoke checks.

### Key synthetic outcomes
- Logistic baseline: Val AUC ~ 0.5901, Test AUC ~ 0.5331, Test proxy EV/trade ~ 0.0123.
- Image CNN (short smoke): Val/Test AUC ~ 0.5000.
- GBDT and GRU tested in ablation context; mixed results, no strong edge signal.
- Readiness report on synthetic ablation produced **yellow** (some checks passed, not all).

## 5) Known limitations / blockers

- Real exchange fetch failed in this environment at least once due network reachability.
- Synthetic results are useful for pipeline sanity only; they do not validate market edge.

## 6) Exact next actions (resume plan)

### Priority A — first true go/no-go (real data)
1. In Colab/networked environment, run:
   - `fetch_ohlcv.py` (Coinbase BTC/USD 5m),
   - `build_windows.py`,
   - `validate_dataset.py`.
2. Freeze dataset artifacts as `dataset_v1` (manifest + split boundaries).
3. Run ablation pipeline:
   - `run_ablation_pipeline.py`.
4. Run:
   - `compare_models.py`,
   - `readiness_report.py`,
   - `backtest_from_model.py` on top baseline.

### Priority B — only if Priority A looks promising
5. Tune thresholds/cost assumptions with strict validation discipline.
6. Improve sequence model before investing more in CNN.
7. Keep CNN only if it beats baselines consistently on test + walk-forward.

## 7) Practical command starter pack (real data)

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

python src/data/validate_dataset.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --windows data/interim/windows/window_index.parquet \
  --timeframe-minutes 5

python src/eval/run_ablation_pipeline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --images-dir data/processed/images_224 \
  --windows data/interim/windows/window_index.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --out-root artifacts/ablation

python src/eval/readiness_report.py \
  --comparison artifacts/ablation/model_comparison.json \
  --min-test-auc 0.55 \
  --min-test-balanced-accuracy 0.53 \
  --min-proxy-ev 0.0 \
  --out artifacts/ablation/readiness_report.json
```

## 8) Resume note for future sessions

If context is missing next time, start with:
1. `HANDOFF_REPORT.md` (this file),
2. `CURRENT_STATE_BRIEF.md`,
3. `ROADMAP_V2.md`,
4. `README.md` command sections.

These four files are enough to quickly recover full context and continue productively.
