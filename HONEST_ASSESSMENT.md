# Honest Assessment: Current Status, Results, and Viability

## 1) What has been done so far

Implemented a full research scaffold for BTC 5-minute next-candle modeling:

- Data layer:
  - `fetch_ohlcv.py` (live exchange ingestion)
  - `generate_synthetic_ohlcv.py` (offline synthetic generation)
  - `build_windows.py` (window/sample index + next-candle linkage)
  - `build_labels.py` (fixed-horizon ternary + triple-barrier labels)
  - `render_charts.py` (deterministic candlestick images)
  - `validate_dataset.py` (integrity checks)
- Model layer:
  - logistic baseline (`train_baseline.py`)
  - GBDT baseline (`train_gbdt_baseline.py`)
  - image CNN (`train_image_cnn.py`)
  - GRU sequence baseline (`train_sequence_model.py`)
- Evaluation layer:
  - walk-forward harness (`eval/walk_forward.py`)
- Documentation/configuration:
  - setup guides, runbooks, roadmap docs, and config presets.

## 2) What results we currently have

All meaningful runs so far were on **synthetic/offline smoke data** (not real exchange data in this environment).

### Previously recorded smoke metrics (synthetic)
- Logistic baseline:
  - Val AUC ~ 0.5901
  - Test AUC ~ 0.5331
- Image CNN (short smoke run):
  - Val/Test AUC ~ 0.5000

### Additional recent smoke runs (synthetic)
- Walk-forward logistic aggregate (3 folds):
  - Mean AUC ~ 0.5268
  - Mean balanced accuracy ~ 0.5214
- GBDT baseline:
  - Val AUC ~ 0.5343
  - Test AUC ~ 0.4885
- GRU sequence baseline (2-epoch smoke):
  - Val AUC ~ 0.4419
  - Test AUC ~ 0.4623

## 3) Implications for further actions

- The pipeline/tooling is **working** and supports multiple model families.
- Current metric values are **not evidence of real trading edge**, because they are from synthetic data and very short smoke runs.
- The only strong conclusion right now:
  - engineering scaffold is ready,
  - model quality remains unproven on real market data.

## 4) Honest viability assessment

### Technical viability: **High**
- It is fully feasible to build and run this system end-to-end.
- Core data/model/eval components are in place.

### Research viability (finding stable predictive edge): **Uncertain / Moderate-Low**
- At 5-minute horizon, BTC is noisy and edge is usually small.
- After realistic fees/slippage, many apparent signals disappear.
- Without rigorous real-data walk-forward validation, there is high risk of overfitting.

### Practical viability for a useful app: **Moderate**, with conditions
Viable if all are true:
1. Real-data out-of-sample performance exceeds strong baselines consistently.
2. Decision metrics remain positive after conservative cost assumptions.
3. Results hold across regimes (not one lucky period).

If these fail, the project is still useful as an ML/data engineering portfolio artifact, but not as a robust live decision engine.

## 5) Recommended immediate next action

Run the exact same pipeline on real Coinbase BTC/USD 5m in Colab/networked environment and treat that as the first true go/no-go checkpoint.
