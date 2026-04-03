# Model Priority (Current Plan)

## Is CNN the main focus?

No.

Current priority order is:
1. Numeric baselines (logistic, GBDT)
2. Sequence baseline (GRU/LSTM-style)
3. Image CNN (only if it adds incremental value)
4. Hybrid model (optional, after ablation evidence)

## Why CNN is not first

- CNN adds more complexity and compute cost.
- In current smoke runs, CNN did not outperform baseline.
- We need a strong real-data baseline benchmark before investing in heavier image pipelines.

## What CNN is used for right now

- Smoke testing pipeline integrity.
- Optional ablation branch.
- Candidate for later improvement if real-data experiments show potential uplift.

## Decision rule

Keep CNN as a main path only if it consistently beats numeric/sequence baselines on:
- out-of-sample AUC / balanced accuracy,
- walk-forward stability,
- and cost-aware decision metrics.
