# Next Action Plan (Post-Setup)

This plan assumes the current repo state is ready (scripts + configs) and the goal is to move from setup to a **validated, tradable research result**.

## Phase 1 — Run the full pipeline once in Colab Free (Day 1)

1. Open Colab, clone repo, install dependencies from `requirements.txt`.
2. Run data build scripts in order:
   - `src/data/fetch_ohlcv.py`
   - `src/data/build_windows.py`
   - `src/data/render_charts.py`
3. Run `src/data/validate_dataset.py` and block progress if any integrity checks fail.
4. Save all generated artifacts to Drive (`data/` + `artifacts/`).

**Exit criteria**
- No duplicate timestamps.
- Gap checks acceptable.
- Split ordering valid.
- Labels aligned (`label_idx == end_idx + 1`).

---

## Phase 2 — Establish baseline performance floor (Day 1–2)

1. Train baseline model with `src/train/train_baseline.py` using current defaults:
   - Coinbase BTC/USD
   - 5m timeframe
   - decision cost = 0.02
   - min edge = 0.03
2. Archive `artifacts/baseline/metrics.json` and model file.
3. Record key metrics in a simple experiment table:
   - AUC, F1, balanced accuracy, Brier
   - decision coverage, hit rate, proxy EV/trade

**Exit criteria**
- Baseline run completed end-to-end.
- Metrics stored and reproducible.

---

## Phase 3 — Train and compare image CNN (Day 2)

1. Train `src/train/train_image_cnn.py` on rendered images.
2. Store `artifacts/image_cnn/metrics.json` + checkpoint.
3. Compare CNN vs baseline on validation and test.

**Decision rule**
- Continue only if CNN improves out-of-sample metrics and decision-level proxy EV.
- If not, iterate on data quality/features before bigger models.

---

## Phase 4 — Improve model quality with low-cost iterations (Day 3–4)

Priority order for Colab Free:
1. Hyperparameter sweep (small grid):
   - baseline: `decision_cost`, `min_edge`
   - CNN: epochs (6–12), lr (1e-4 to 3e-3), batch size (32/64)
2. Add image augmentations (light only): random brightness/contrast, minor affine.
3. Add class balancing if split imbalance is significant.
4. Repeat evaluation with the same fixed split boundaries.

**Exit criteria**
- At least one model variant beats initial baseline robustly on test.

---

## Phase 5 — Align with Polymarket execution logic (Day 4)

1. Convert model probability into action rules:
   - UP bet, DOWN bet, or SKIP.
2. Replace proxy cost with a more realistic fee/slippage assumption when known.
3. Recompute decision metrics under those assumptions.

**Exit criteria**
- Action policy produces stable coverage and positive proxy EV over test period.

---

## Phase 6 — Risk and robustness checks (Day 5)

1. Time-sliced performance:
   - month-by-month
   - high-volatility vs low-volatility periods
2. Drawdown and losing-streak diagnostics for decision sequence.
3. Fail-safe thresholds:
   - max consecutive losses
   - min confidence edge required to place bet

**Exit criteria**
- No catastrophic regime failure in held-out test segments.

---

## Phase 7 — Package for repeatable daily use (Day 5–6)

1. Create one Colab notebook to run:
   - fetch latest data,
   - refresh windows/images,
   - load best model,
   - output next-candle UP/DOWN/SKIP decision.
2. Save daily prediction logs to Drive/CSV.
3. Add a manual checklist before acting on predictions.

**Exit criteria**
- One-click reproducible inference workflow with logged decisions.

---

## Immediate next 3 commands to execute first

```bash
python src/data/validate_dataset.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --windows data/interim/windows/window_index.parquet \
  --timeframe-minutes 5

python src/train/train_baseline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --decision-cost 0.02 \
  --min-edge 0.03 \
  --out-dir artifacts/baseline

python src/train/train_image_cnn.py \
  --images-dir data/processed/images_224 \
  --windows data/interim/windows/window_index.parquet \
  --epochs 8 \
  --batch-size 64 \
  --lr 1e-3 \
  --out-dir artifacts/image_cnn
```
