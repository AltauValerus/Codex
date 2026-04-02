# Run Results (Executed in this environment)

## What was executed

1. Installed dependencies from `requirements.txt`.
2. Attempted real exchange fetch via Coinbase (`src/data/fetch_ohlcv.py`) — failed due network unreachable.
3. Executed offline smoke path with synthetic data:
   - `src/data/generate_synthetic_ohlcv.py`
   - `src/data/build_windows.py`
   - `src/data/render_charts.py`
   - `src/data/validate_dataset.py`
   - `src/train/train_baseline.py`
   - `src/train/train_image_cnn.py`

## Key outcomes

### Baseline (synthetic)
- Val AUC: `0.5901`
- Test AUC: `0.5331`
- Test decision coverage: `0.3470`
- Test proxy EV/trade: `0.0123`

Artifacts:
- `artifacts/baseline_synth/baseline_direction_model.joblib`
- `artifacts/baseline_synth/metrics.json`

### Image CNN (synthetic, 2 epochs, CPU)
- Val AUC: `0.5000`
- Test AUC: `0.5000`

Artifacts:
- `artifacts/image_cnn_synth/image_cnn.pt`
- `artifacts/image_cnn_synth/metrics.json`

## Interpretation

- Pipeline is operational end-to-end.
- Baseline outperformed chance on synthetic data.
- CNN run is currently a smoke test (small synthetic dataset + short training) and not yet a meaningful benchmark.

## Next practical move

Run the same flow on real market data in Colab with working internet access and compare real out-of-sample metrics.
