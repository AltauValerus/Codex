# Architecture Review (Current State)

## Verdict

The overall direction is solid and practical for research-stage work.

The pipeline has a correct high-level structure:
- deterministic data prep,
- strict time-based splits,
- multiple baseline/model families,
- and evaluation/reporting layers.

## Key flaws / risks identified

1. **Feature logic is duplicated in multiple scripts**
   - Similar feature engineering blocks are repeated in baseline, GBDT, sequence, walk-forward, and backtest scripts.
   - Risk: silent training/inference mismatch over time.

2. **Backtest payoff model is simplified**
   - Current decision PnL model is intentionally lightweight.
   - Risk: over/under-estimating practical edge vs real market execution rules.

3. **No unified experiment metadata schema**
   - Metrics files are similar but not fully standardized across all models.
   - Risk: harder automation, harder apples-to-apples comparisons.

4. **Network dependency still blocks real data phase in this environment**
   - Architecture supports it, but execution is environment-constrained.

## Recommended next step (move forward)

Proceed with the next implementation step while addressing the top risk:

1. Refactor shared feature engineering into a single reusable module.
2. Update all train/eval/backtest scripts to import shared feature builder.
3. Re-run ablation pipeline and ensure no metric schema regressions.

This preserves momentum and reduces architecture drift risk.
