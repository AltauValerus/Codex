# Roadmap V2 — Next Actions (Incorporating Research + Current Repo State)

This roadmap builds on what is already implemented in this repository and aligns execution with leakage-resistant finance ML practices.

## Current baseline (already done)

- End-to-end pipeline exists: OHLCV ingest/generate -> window labels -> chart render -> validation -> baseline/CNN train.
- Offline smoke run completed on synthetic data.
- Real exchange fetch was attempted but failed in this environment due network reachability.

---

## Phase 0 (Immediate): Run on real data in Colab and freeze a reproducible dataset snapshot

### Goals
1. Execute the existing pipeline on **real Coinbase BTC/USD 5m data** in Colab.
2. Produce immutable `dataset_v1` artifacts and a manifest.
3. Keep train/val/test date boundaries fixed for all subsequent experiments.

### Actions
- Run `fetch_ohlcv.py` in Colab with paginated backfill and closed-candle only policy.
- Run `build_windows.py` with finalized split cutoffs.
- Run `validate_dataset.py` and fail fast on:
  - duplicates,
  - nonstandard gaps,
  - split-order failures,
  - label misalignment.
- Persist:
  - `data/raw/ohlcv/*.parquet`
  - `data/interim/windows/*.parquet`
  - `data/processed/images_224/*`
  - metadata file with source, timeframe, timezone, split boundaries, and hash.

### Exit criteria
- `dataset_v1` reproducible from manifest.
- Validation report clean or documented with approved gap policy.

---

## Phase 1: Strengthen labeling and evaluation protocol (before larger models)

### Goals
- Move from a single binary label to robust alternatives.
- Prevent over-optimistic estimates via time-series methodology.

### Actions
1. Add **ternary fixed-horizon labels** (`up/flat/down`) with configurable threshold.
2. Add **triple-barrier labels** (profit-take/stop-loss/timeout).
3. Implement walk-forward evaluation harness:
   - expanding train window,
   - fixed test windows,
   - fold aggregation.
4. Add purge/embargo option for overlapping event labels.

### Deliverables
- `src/data/build_labels.py` (or extension of current label builder)
- `src/eval/walk_forward.py`
- `artifacts/eval/labeling_comparison.json`

### Exit criteria
- One label design selected based on stability across folds/regimes, not single-split best score.

---

## Phase 2: Make baseline stack truly competitive

### Goals
- Ensure deep models are compared against hard-to-beat numeric baselines.

### Actions
1. Keep current logistic baseline.
2. Add gradient boosting baseline (LightGBM/XGBoost or sklearn HistGB).
3. Add calibration comparison (Platt vs isotonic).
4. Tune decision layer (`cost`, `min_edge`) on validation only.

### Deliverables
- `src/train/train_gbdt_baseline.py`
- unified metrics schema (`auc`, `f1`, `balanced_acc`, calibration, decision coverage, proxy EV)

### Exit criteria
- Best baseline is strong and reproducible; deep models must beat it out-of-sample.

---

## Phase 3: Sequence model first, then image model justification

### Goals
- Follow evidence-driven ordering: numeric/sequence before image-heavy architectures.

### Actions
1. Add compact sequence model on candle features (GRU/LSTM first).
2. Run ablation:
   - baseline vs sequence,
   - with/without indicator set,
   - different lookback lengths.
3. Keep current image CNN as optional path and require incremental lift proof.

### Deliverables
- `src/train/train_sequence_model.py`
- `artifacts/ablation/sequence_vs_baseline.json`
- optional `artifacts/ablation/image_vs_sequence.json`

### Exit criteria
- Sequence model shows stable multi-fold gains OR image pipeline demonstrates measurable incremental value.

---

## Phase 4: Trading-aligned evaluation and risk controls

### Goals
- Convert prediction quality into action quality under realistic assumptions.

### Actions
1. Add backtest module with:
   - transaction costs,
   - slippage scenarios,
   - action states (UP / DOWN / SKIP),
   - drawdown/turnover tracking.
2. Add stress slices:
   - high-volatility regime,
   - low-volatility regime,
   - month-by-month performance.
3. Report degradation from validation to test and from backtest to paper/live.

### Deliverables
- `src/eval/backtest.py`
- `artifacts/eval/backtest_summary.json`
- `artifacts/eval/regime_breakdown.json`

### Exit criteria
- Strategy remains acceptable after costs and across regimes; no catastrophic fold.

---

## Phase 5: App integration and MLOps hardening

### Goals
- Turn research pipeline into app-ready, monitored inference service.

### Actions
1. Define strict feature contract (offline/online parity).
2. Export chosen model (ONNX or TorchScript) and version in registry metadata.
3. Build inference service endpoint returning:
   - class probs,
   - action recommendation,
   - confidence and model version.
4. Monitoring:
   - data quality alerts (missing candles, timestamp jumps),
   - prediction drift,
   - calibration drift,
   - live hit-rate tracking.

### Deliverables
- `src/serve/inference_service.py`
- `docs/model_contract.md`
- `docs/monitoring_spec.md`

### Exit criteria
- Stable paper-trading period with logged decisions and monitored drift.

---

## Priority order for the next 7 days

Day 1-2:
- Execute real-data Colab run and freeze `dataset_v1`.
- Produce baseline + validation report on real data.

Day 3-4:
- Add walk-forward harness + ternary/triple-barrier labeling.
- Re-run baselines under new protocol.

Day 5:
- Add sequence model and ablations.

Day 6:
- Add backtesting + regime diagnostics.

Day 7:
- Choose final candidate model, define app inference contract, start paper-trading mode.

---

## Decision gates (must pass)

1. **Data gate:** validation checks pass and dataset is reproducible.
2. **Model gate:** candidate outperforms best baseline across folds (not one split).
3. **Risk gate:** acceptable drawdown and stable regime behavior after costs.
4. **Deployment gate:** inference contract + monitoring implemented before user-facing rollout.
