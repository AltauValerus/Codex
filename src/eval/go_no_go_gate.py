"""Generate final go/no-go recommendation from readiness + backtest outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--readiness", required=True)
    p.add_argument("--backtest", required=True)
    p.add_argument("--manifest", required=False, default=None)
    p.add_argument("--gate-model-name", required=False, default=None)
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--min-hit-rate", type=float, default=0.52)
    p.add_argument("--max-dd", type=float, default=-0.25, help="Maximum allowed drawdown pct (negative)")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    readiness = json.loads(Path(args.readiness).read_text())
    backtest = json.loads(Path(args.backtest).read_text())
    manifest = json.loads(Path(args.manifest).read_text()) if args.manifest else None

    readiness_status = readiness.get("readiness", {}).get("status")
    summary = backtest.get("summary", {})

    n_trades = int(summary.get("n_trades", 0))
    hit_rate = float(summary.get("hit_rate", 0.0))
    max_dd = float(summary.get("max_drawdown_pct", 0.0))
    avg_pnl = float(summary.get("avg_pnl_per_trade", 0.0))

    checks = {
        "readiness_not_red": readiness_status in {"green", "yellow"},
        "min_trades": n_trades >= args.min_trades,
        "min_hit_rate": hit_rate >= args.min_hit_rate,
        "max_drawdown": max_dd >= args.max_dd,
        "avg_pnl_positive": avg_pnl > 0,
    }

    passed = sum(bool(v) for v in checks.values())

    if readiness_status == "green" and passed >= 4:
        decision = "go_paper"
        rationale = "Strong enough for paper-trading stage with monitoring."
    elif readiness_status in {"green", "yellow"} and passed >= 3:
        decision = "iterate"
        rationale = "Promising but not strong enough; iterate and re-run gate."
    else:
        decision = "no_go"
        rationale = "Insufficient evidence; do not advance beyond research loop."

    report = {
        "decision": decision,
        "rationale": rationale,
        "checks": checks,
        "inputs": {
            "readiness": args.readiness,
            "backtest": args.backtest,
            "manifest": args.manifest,
            "thresholds": {
                "min_trades": args.min_trades,
                "min_hit_rate": args.min_hit_rate,
                "max_dd": args.max_dd,
            },
        },
        "context": {
            "readiness_status": readiness_status,
            "n_trades": n_trades,
            "hit_rate": hit_rate,
            "max_drawdown_pct": max_dd,
            "avg_pnl_per_trade": avg_pnl,
            "dataset_name": manifest.get("dataset_name") if manifest else None,
            "gate_model_name": args.gate_model_name,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Saved go/no-go report to {out}")


if __name__ == "__main__":
    main()
