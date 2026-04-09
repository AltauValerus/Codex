# Simple Mode (No Overengineering)

You are right: if your real usage is one model, you do NOT need to run full multi-model benchmarking every time.

## Practical recommendation

Use a two-speed workflow:

1. **Setup/selection phase (occasional):**
   - compare a few models (baseline, GBDT, sequence) once in a while.

2. **Operational phase (default):**
   - run only one selected model pipeline repeatedly.

## Default single-model path

For simplicity and stability, start with the logistic baseline path and improve thresholds/cost assumptions over time.

Use:
- `python -m src.eval.run_single_model_pipeline`

This keeps the process lean while preserving risk controls (backtest + go/no-go report).
