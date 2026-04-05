# Current State Brief

## What has been done so far

1. Built a full research pipeline for 5-minute BTC next-candle prediction:
   - data fetch/generation,
   - label construction,
   - deterministic chart-image creation,
   - data validation,
   - baseline and deep model training,
   - walk-forward and model-comparison evaluation.

2. Added model families and comparison flow:
   - Logistic baseline,
   - GBDT baseline,
   - GRU sequence model,
   - Image CNN,
   - comparison utility + one-command ablation runner.

3. Added clear runbooks and planning docs:
   - roadmap, next-step docs, honest assessment, run results, and README command guides.

## What we currently know

- Engineering side: pipeline is functional end-to-end.
- Research side: robust edge is not proven yet.
- Why: meaningful results are still mostly synthetic/offline smoke checks in this environment.

## Current understanding of the task

The task is **not** to deploy live immediately.
It is to run disciplined, leakage-resistant validation on real BTC/USD 5m data and determine if there is a stable, cost-aware predictive signal.

In practical terms, the objective is:
1. Build/validate a reproducible real dataset (`dataset_v1`).
2. Compare baselines and richer models fairly (same splits/protocol).
3. Keep only models that remain strong out-of-sample after costs.
4. Move to paper-trading only after those checks pass.

## Immediate next checkpoint

Run the full pipeline on real Coinbase BTC/USD 5m in Colab/networked environment and use that as the first real go/no-go decision point.
