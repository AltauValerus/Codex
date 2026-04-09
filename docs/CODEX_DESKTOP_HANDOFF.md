# Codex Desktop Handoff

## Objective

Continue the BTC/USD 5-minute next-candle classification project from the new **numeric feature expansion + controlled experiment sweep** phase.

The current target is:
- keep the existing real-data leakage/gap-safety fixes,
- expand the numeric feature space,
- export split-level predictions,
- select `min_edge` on validation only,
- compare models with realized-return and regime-aware reporting,
- promote a new champion only if it actually beats the current GBDT incumbent.

## What Was Implemented

### Shared feature and policy layer

- `src/features/common.py`
  - kept `feature_set="baseline"` backward-compatible,
  - added `feature_set="v2_numeric"`,
  - added prediction-context helpers:
    - `next_bar_return`
    - `realized_vol_24`
    - `volume_ratio_24`
    - `volume_z_24_context`

- `src/eval/policy_utils.py`
  - added shared classification metrics,
  - added split-level prediction export builder,
  - added proxy + realized decision-policy reporting,
  - added regime slices:
    - monthly
    - realized-volatility terciles
    - high-volume vs low-volume
  - added validation-only `min_edge` sweep logic.

### Trainers updated

- `src/train/train_baseline.py`
- `src/train/train_gbdt_baseline.py`
- `src/train/train_sequence_model.py`

Each now supports:
- `--feature-set`
- split-level prediction export to parquet
- optional validation-only threshold sweep
- richer metrics JSON including:
  - selected threshold
  - proxy PnL
  - realized next-bar return PnL
  - drawdown
  - regime slices

### Evaluation updated

- `src/eval/walk_forward.py`
  - now supports `--feature-set`

- `src/eval/backtest_from_model.py`
  - now supports `--feature-set`
  - now reports both proxy and realized next-bar return backtests

- `src/eval/compare_models.py`
  - flattened report now includes:
    - `feature_set`
    - `calibration`
    - `selected_min_edge`
    - coverage
    - hit rate
    - proxy PnL
    - realized PnL
    - drawdown

### New orchestration utilities

- `src/eval/run_numeric_experiments.py`
  - fixed experiment matrix for:
    - logistic baseline/v2
    - GBDT baseline/v2
    - GBDT v2 hyperparameter sweep
    - GRU v2 lookback/hidden-dim sweep
  - incumbent-aware champion selection

- `src/eval/sweep_decision_threshold.py`
  - standalone validation-threshold selection utility

## What Was Verified Today

### Static verification

Passed:

```bash
python -m py_compile \
  src/features/common.py \
  src/eval/policy_utils.py \
  src/eval/compare_models.py \
  src/eval/walk_forward.py \
  src/eval/backtest_from_model.py \
  src/eval/run_numeric_experiments.py \
  src/eval/sweep_decision_threshold.py \
  src/train/train_baseline.py \
  src/train/train_gbdt_baseline.py \
  src/train/train_sequence_model.py
```

### Full-data regression check

Ran:

```bash
.venv\Scripts\python.exe src/train/train_baseline.py ^
  --ohlcv data\raw\ohlcv\coinbase_btcusd_5m.parquet ^
  --train-end 2024-06-30T23:55:00Z ^
  --val-end 2025-03-31T23:55:00Z ^
  --decision-cost 0.02 ^
  --min-edge 0.03 ^
  --out-dir artifacts\regression_baseline_full
```

Result:
- test AUC: `0.5177882564`
- test balanced accuracy: `0.5110262582`

This matches the prior baseline almost exactly, so the new plumbing did not materially change the old baseline behavior.

### Real-data smoke run on richer features

Created subset:

```bash
data/raw/ohlcv/coinbase_btcusd_5m_smoke.parquet
```

Range used:
- `2024-01-01` through `2024-03-31`

Ran:

```bash
.venv\Scripts\python.exe src/train/train_gbdt_baseline.py ^
  --ohlcv data\raw\ohlcv\coinbase_btcusd_5m_smoke.parquet ^
  --train-end 2024-02-15T23:55:00Z ^
  --val-end 2024-03-10T23:55:00Z ^
  --decision-cost 0.02 ^
  --feature-set v2_numeric ^
  --calibration isotonic ^
  --sweep-min-edge ^
  --out-dir artifacts\smoke_gbdt_v2
```

Result:
- selected `min_edge`: `0.04`
- val AUC: `0.5116556262`
- test AUC: `0.5193947587`
- test balanced accuracy: `0.5130555185`
- test coverage: `0.0658177609`
- test hit rate: `0.5552763819`

Prediction exports were created successfully for train/val/test.

## What Is Not Finished Yet

- logistic `v2_numeric` smoke run not executed yet
- GRU `v2_numeric` smoke run not executed yet
- `backtest_from_model.py` was updated but not yet smoke-tested after the refactor
- `compare_models.py` updated but not yet exercised on the new smoke outputs
- `run_numeric_experiments.py` was implemented but **not** run end-to-end yet
- no PR has been created from this local clone yet

## Best Resume Sequence For Tomorrow

### 1. Quick sanity check

```bash
python -m py_compile src/features/common.py src/eval/policy_utils.py src/eval/compare_models.py src/eval/walk_forward.py src/eval/backtest_from_model.py src/eval/run_numeric_experiments.py src/eval/sweep_decision_threshold.py src/train/train_baseline.py src/train/train_gbdt_baseline.py src/train/train_sequence_model.py
```

### 2. Finish smoke runs

Run logistic `v2_numeric` on the smoke subset.

Run GRU `v2_numeric` on the smoke subset with something small first, e.g.:

```bash
.venv\Scripts\python.exe src/train/train_sequence_model.py ^
  --ohlcv data\raw\ohlcv\coinbase_btcusd_5m_smoke.parquet ^
  --train-end 2024-02-15T23:55:00Z ^
  --val-end 2024-03-10T23:55:00Z ^
  --feature-set v2_numeric ^
  --lookback 48 ^
  --hidden-dim 64 ^
  --epochs 2 ^
  --sweep-min-edge ^
  --out-dir artifacts\smoke_gru_v2
```

### 3. Smoke-test the updated backtest script

```bash
.venv\Scripts\python.exe src/eval/backtest_from_model.py ^
  --model artifacts\smoke_gbdt_v2\gbdt_direction_model.joblib ^
  --ohlcv data\raw\ohlcv\coinbase_btcusd_5m_smoke.parquet ^
  --val-end 2024-03-10T23:55:00Z ^
  --feature-set v2_numeric ^
  --min-edge 0.04 ^
  --decision-cost 0.02 ^
  --out artifacts\smoke_gbdt_v2\backtest.json
```

### 4. Exercise `compare_models.py`

Use the smoke outputs first before running the full matrix.

### 5. Only then run the full experiment matrix

```bash
.venv\Scripts\python.exe src/eval/run_numeric_experiments.py ^
  --ohlcv data\raw\ohlcv\coinbase_btcusd_5m.parquet ^
  --train-end 2024-06-30T23:55:00Z ^
  --val-end 2025-03-31T23:55:00Z ^
  --decision-cost 0.02 ^
  --out-dir artifacts\numeric_experiments_real
```

## Important Notes

- The current realized-return backtest uses raw next-bar return minus `decision_cost` and `slippage`.
- With `decision_cost=0.02`, realized returns are expected to stay heavily negative on 5-minute BTC bars.
- That is consistent with the current placeholder cost assumption, but it is probably too punitive for practical ranking.
- Do **not** silently change it without noting the assumption explicitly.

- `data/` and `artifacts/` are ignored by git, so the code commit will not preserve the generated parquet/json outputs.
- The important persistent part is the code and this handoff file.
