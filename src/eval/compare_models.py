"""Aggregate and compare model metrics into a single table."""

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
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def schema_warnings(metrics: dict) -> list[str]:
    warnings = []
    required_paths = [
        ("val", "auc"),
        ("test", "auc"),
        ("val", "balanced_accuracy"),
        ("test", "balanced_accuracy"),
        ("val", "decision", "coverage"),
        ("test", "decision", "coverage"),
        ("val", "decision", "realized_total_pnl"),
        ("test", "decision", "realized_total_pnl"),
    ]
    for path in required_paths:
        if pick(metrics, *path) is None:
            warnings.append(f"missing:{'.'.join(path)}")
    if pick(metrics, "samples") is None and pick(metrics, "counts") is None:
        warnings.append("missing:samples_or_counts")
    return warnings


def flatten_metrics(name: str, metrics: dict) -> dict:
    return {
        "model_name": name,
        "reported_model_name": metrics.get("model_name"),
        "feature_set": metrics.get("feature_set"),
        "calibration": metrics.get("calibration"),
        "selected_min_edge": metrics.get("selected_min_edge"),
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
        "val_hit_rate": pick(metrics, "val", "decision", "hit_rate"),
        "test_hit_rate": pick(metrics, "test", "decision", "hit_rate"),
        "val_proxy_ev_per_trade": pick(metrics, "val", "decision", "proxy_ev_per_trade"),
        "test_proxy_ev_per_trade": pick(metrics, "test", "decision", "proxy_ev_per_trade"),
        "val_proxy_total_pnl": pick(metrics, "val", "decision", "proxy_total_pnl"),
        "test_proxy_total_pnl": pick(metrics, "test", "decision", "proxy_total_pnl"),
        "val_proxy_max_drawdown_pct": pick(metrics, "val", "decision", "proxy_max_drawdown_pct"),
        "test_proxy_max_drawdown_pct": pick(metrics, "test", "decision", "proxy_max_drawdown_pct"),
        "val_realized_ev_per_trade": pick(metrics, "val", "decision", "realized_ev_per_trade"),
        "test_realized_ev_per_trade": pick(metrics, "test", "decision", "realized_ev_per_trade"),
        "val_realized_total_pnl": pick(metrics, "val", "decision", "realized_total_pnl"),
        "test_realized_total_pnl": pick(metrics, "test", "decision", "realized_total_pnl"),
        "val_realized_max_drawdown_pct": pick(metrics, "val", "decision", "realized_max_drawdown_pct"),
        "test_realized_max_drawdown_pct": pick(metrics, "test", "decision", "realized_max_drawdown_pct"),
        "samples_train": pick(metrics, "samples", "train", default=pick(metrics, "counts", "train")),
        "samples_val": pick(metrics, "samples", "val", default=pick(metrics, "counts", "val")),
        "samples_test": pick(metrics, "samples", "test", default=pick(metrics, "counts", "test")),
    }


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
        row = flatten_metrics(name, metrics)
        row["schema_warnings"] = schema_warnings(metrics)
        rows.append(row)

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
                writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
                writer.writeheader()
                writer.writerows(rows_sorted)

    print(json.dumps(report, indent=2))
    print(f"Saved comparison to {out}")


if __name__ == "__main__":
    main()
