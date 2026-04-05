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
