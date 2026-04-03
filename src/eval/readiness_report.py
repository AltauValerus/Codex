"""Create a business-friendly readiness report from model comparison output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--comparison", required=True, help="Path to model_comparison.json")
    p.add_argument("--min-test-auc", type=float, default=0.55)
    p.add_argument("--min-test-balanced-accuracy", type=float, default=0.53)
    p.add_argument("--min-proxy-ev", type=float, default=0.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def evaluate_top_model(rows: list[dict], min_auc: float, min_bal_acc: float, min_ev: float) -> dict:
    if not rows:
        return {
            "status": "red",
            "reason": "No models were available in comparison report.",
            "top_model": None,
            "checks": {},
        }

    top = rows[0]
    auc = top.get("test_auc")
    bal = top.get("test_balanced_accuracy")
    ev = top.get("test_proxy_ev_per_trade")

    checks = {
        "test_auc": {"value": auc, "threshold": min_auc, "pass": auc is not None and auc >= min_auc},
        "test_balanced_accuracy": {
            "value": bal,
            "threshold": min_bal_acc,
            "pass": bal is not None and bal >= min_bal_acc,
        },
        "test_proxy_ev_per_trade": {
            "value": ev,
            "threshold": min_ev,
            "pass": ev is not None and ev >= min_ev,
        },
    }

    passed = sum(1 for c in checks.values() if c["pass"])

    if passed == 3:
        status = "green"
        reason = "Top model passes all readiness thresholds."
    elif passed >= 1:
        status = "yellow"
        reason = "Top model passes some thresholds; iterate before paper/live decisions."
    else:
        status = "red"
        reason = "Top model fails all thresholds; not ready beyond research iteration."

    return {
        "status": status,
        "reason": reason,
        "top_model": top.get("model_name"),
        "checks": checks,
    }


def main() -> None:
    args = parse_args()

    with Path(args.comparison).open("r", encoding="utf-8") as f:
        comp = json.load(f)

    rows = comp.get("rows", [])
    readiness = evaluate_top_model(rows, args.min_test_auc, args.min_test_balanced_accuracy, args.min_proxy_ev)

    out_report = {
        "input": {
            "comparison": args.comparison,
            "thresholds": {
                "min_test_auc": args.min_test_auc,
                "min_test_balanced_accuracy": args.min_test_balanced_accuracy,
                "min_proxy_ev": args.min_proxy_ev,
            },
        },
        "readiness": readiness,
        "n_models_loaded": comp.get("n_models_loaded"),
        "best_by_test_auc": comp.get("best_by_test_auc"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)

    print(json.dumps(out_report, indent=2))
    print(f"Saved readiness report to {out}")


if __name__ == "__main__":
    main()
