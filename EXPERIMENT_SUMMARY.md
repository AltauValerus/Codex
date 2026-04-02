# Experiment Summary

## What has been completed

- Built an end-to-end BTC next-candle pipeline including:
  - data fetch/prep scripts,
  - deterministic candlestick image rendering,
  - dataset validation,
  - baseline model training,
  - image-CNN training.
- Executed a full **offline smoke run** using synthetic OHLCV because live exchange fetch was unreachable in this environment.

## Image counts used in the executed run

From `window_index_synth.parquet` and rendered image folders:
- **Train:** 241 images
- **Validation:** 144 images
- **Test:** 267 images
- **Total:** 652 images

## Results achieved

### Baseline model (numeric features)
- Validation AUC: **0.5901**
- Validation F1: **0.1609**
- Validation Balanced Accuracy: **0.5366**
- Validation Brier: **0.2511**
- Validation decision coverage: **0.4653**
- Validation proxy EV/trade: **0.0248**

- Test AUC: **0.5331**
- Test F1: **0.1830**
- Test Balanced Accuracy: **0.5098**
- Test Brier: **0.2485**
- Test decision coverage: **0.3470**
- Test proxy EV/trade: **0.0123**

### Image CNN model (2-epoch smoke test)
- Validation AUC: **0.5000**
- Validation F1: **0.7085**
- Validation Balanced Accuracy: **0.5000**

- Test AUC: **0.5000**
- Test F1: **0.6412**
- Test Balanced Accuracy: **0.5000**

Interpretation:
- Baseline is above chance on this synthetic run.
- CNN run is currently a smoke check and not yet competitive.

## Suggested next steps

1. Run the same pipeline on **real Coinbase BTC/USD 5m data** in Colab with working internet.
2. Increase CNN training budget (epochs 15–30, early stopping, lr scheduling).
3. Add augmentations and regularization for image model.
4. Compare image-only vs numeric baseline vs (next) hybrid model.
5. Replace proxy decision-cost assumption with realistic Polymarket execution costs.
6. Perform rolling/walk-forward evaluation and monthly regime diagnostics.
