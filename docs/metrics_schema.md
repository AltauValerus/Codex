# Metrics Schema (Minimum)

Each model `metrics.json` should include:

- `val.auc`
- `test.auc`
- `val.f1`
- `test.f1`
- `val.balanced_accuracy`
- `test.balanced_accuracy`
- `samples` or `counts` with `train/val/test`

Optional but recommended:
- `val.brier`, `test.brier`
- decision metrics under `val.decision` and `test.decision`

## Comparison schema consumed by readiness

`src.eval.readiness_report` expects a comparison payload with:

- `n_models_loaded` (int)
- `n_models_missing` (int)
- `models_missing` (array)
- `best_by_test_auc` (string or null)
- `rows` (array of model summaries sorted by test AUC desc)

Each `rows[]` entry should expose at least:
- `model_name`
- `test_auc`
- `test_balanced_accuracy`
- `test_proxy_ev_per_trade`

Single-model mode generates this schema via:
- `src.eval.build_single_model_comparison`
