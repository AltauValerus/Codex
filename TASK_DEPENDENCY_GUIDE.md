# Task Dependency Guide

## Should tasks run sequentially or in parallel?

For this project, use a **hybrid approach**:

- Some tasks are **strictly sequential** (must finish first).
- Some tasks can run **in parallel** once prerequisites are complete.

## Recommended order

### Stage A (Sequential — do in order)
1. Connectivity preflight
2. Real OHLCV fetch
3. Build windows/labels
4. Dataset validation
5. Freeze dataset manifest (`dataset_v1`)

These are dependencies for reliable downstream modeling.

### Stage B (Parallel possible)
After Stage A is complete, these can run in parallel if resources allow:
- logistic baseline training,
- GBDT baseline training,
- sequence model training,
- image CNN training (if image data is present).

### Stage C (Sequential)
1. Compare model outputs
2. Generate readiness report
3. Run backtest on top candidate
4. Decide go/no-go

These are decision-gate steps and should follow Stage B outputs.

## Practical rule of thumb

- If a task produces an artifact used by another task (data file, model file, metrics file),
  it is a dependency and should be completed first.
- If tasks only read shared frozen data and write independent outputs, they can run in parallel.
