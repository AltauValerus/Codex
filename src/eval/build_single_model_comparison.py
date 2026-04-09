"""Build a comparison-style JSON payload from a single baseline metrics file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True, help="Path to baseline metrics.json")
    p.add_argument("--model-name", default="baseline")
    p.add_argument("--out", required=True, help="Output comparison JSON path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))

    report = {
        "n_models_loaded": 1,
        "n_models_missing": 0,
        "models_missing": [],
        "rows": [
            {
                "model_name": args.model_name,
                "val_auc": metrics.get("val", {}).get("auc"),
                "test_auc": metrics.get("test", {}).get("auc"),
                "val_f1": metrics.get("val", {}).get("f1"),
                "test_f1": metrics.get("test", {}).get("f1"),
                "val_balanced_accuracy": metrics.get("val", {}).get("balanced_accuracy"),
                "test_balanced_accuracy": metrics.get("test", {}).get("balanced_accuracy"),
                "val_proxy_ev_per_trade": (metrics.get("val", {}).get("decision") or {}).get("proxy_ev_per_trade"),
                "test_proxy_ev_per_trade": (metrics.get("test", {}).get("decision") or {}).get("proxy_ev_per_trade"),
            }
        ],
        "best_by_test_auc": args.model_name,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved single-model comparison to {out}")


if __name__ == "__main__":
    main()
