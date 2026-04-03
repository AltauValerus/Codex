"""Aggregate and compare model metrics into a single table.

Example:
python src/eval/compare_models.py \
  --model logistic=artifacts/baseline/metrics.json \
  --model gbdt=artifacts/gbdt_baseline/metrics.json \
  --model gru=artifacts/sequence_gru/metrics.json \
  --model cnn=artifacts/image_cnn/metrics.json \
  --out artifacts/eval/model_comparison.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        action="append",
        required=True,
        help="Format: name=path/to/metrics.json; can be passed multiple times",
    )
    p.add_argument("--out", required=True, help="Output .json path")
    p.add_argument("--out-csv", default=None, help="Optional output .csv path")
    return p.parse_args()


def parse_model_arg(model_arg: str) -> tuple[str, Path]:
    if "=" not in model_arg:
        raise ValueError(f"Invalid --model '{model_arg}'. Expected name=path")
    name, path = model_arg.split("=", 1)
    return name.strip(), Path(path.strip())


def pick(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def flatten_metrics(name: str, metrics: dict) -> dict:
    row = {
        "model_name": name,
        "val_auc": pick(metrics, "val", "auc"),
        "test_auc": pick(metrics, "test", "auc"),
        "val_f1": pick(metrics, "val", "f1"),
        "test_f1": pick(metrics, "test", "f1"),
        "val_balanced_accuracy": pick(metrics, "val", "balanced_accuracy"),
        "test_balanced_accuracy": pick(metrics, "test", "balanced_accuracy"),
        "val_brier": pick(metrics, "val", "brier"),
        "test_brier": pick(metrics, "test", "brier"),
        "val_decision_coverage": pick(metrics, "val", "decision", "coverage"),
        "test_decision_coverage": pick(metrics, "test", "decision", "coverage"),
        "val_proxy_ev_per_trade": pick(metrics, "val", "decision", "proxy_ev_per_trade"),
        "test_proxy_ev_per_trade": pick(metrics, "test", "decision", "proxy_ev_per_trade"),
        "samples_train": pick(metrics, "samples", "train", default=pick(metrics, "counts", "train")),
        "samples_val": pick(metrics, "samples", "val", default=pick(metrics, "counts", "val")),
        "samples_test": pick(metrics, "samples", "test", default=pick(metrics, "counts", "test")),
    }
    return row


def main() -> None:
    args = parse_args()

    rows = []
    missing = []

    for model_arg in args.model:
        name, path = parse_model_arg(model_arg)
        if not path.exists():
            missing.append({"model_name": name, "path": str(path), "status": "missing"})
            continue

        with path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append(flatten_metrics(name, metrics))

    rows_sorted = sorted(rows, key=lambda r: (r.get("test_auc") is None, -(r.get("test_auc") or -1)))

    report = {
        "n_models_loaded": len(rows_sorted),
        "n_models_missing": len(missing),
        "models_missing": missing,
        "rows": rows_sorted,
        "best_by_test_auc": rows_sorted[0]["model_name"] if rows_sorted else None,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if rows_sorted:
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
                w.writeheader()
                w.writerows(rows_sorted)

    print(json.dumps(report, indent=2))
    print(f"Saved comparison to {out}")


if __name__ == "__main__":
    main()
