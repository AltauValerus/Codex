"""Run the fixed numeric experiment matrix and select a candidate champion."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[2]))

from src.eval.compare_models import flatten_metrics, schema_warnings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--decision-cost", type=float, default=0.02)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--sweep-start", type=float, default=0.0)
    p.add_argument("--sweep-end", type=float, default=0.08)
    p.add_argument("--sweep-step", type=float, default=0.01)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def selection_key(row: dict, min_coverage: float) -> tuple[float, float, float, float, float]:
    coverage = row.get("val_decision_coverage") or 0.0
    realized_total = row.get("val_realized_total_pnl") or float("-inf")
    auc = row.get("val_auc") or float("-inf")
    drawdown = row.get("val_realized_max_drawdown_pct") or float("-inf")
    return (
        1.0 if coverage >= min_coverage else 0.0,
        1.0 if realized_total > 0.0 else 0.0,
        realized_total,
        auc,
        drawdown,
    )


def beats_incumbent(candidate: dict, incumbent: dict) -> dict[str, bool]:
    cand_auc = candidate.get("test_auc") or float("-inf")
    inc_auc = incumbent.get("test_auc") or float("-inf")
    cand_realized = candidate.get("test_realized_total_pnl") or float("-inf")
    inc_realized = incumbent.get("test_realized_total_pnl") or float("-inf")
    cand_drawdown = candidate.get("test_realized_max_drawdown_pct") or float("-inf")
    inc_drawdown = incumbent.get("test_realized_max_drawdown_pct") or float("-inf")
    cand_hit = candidate.get("test_hit_rate") or float("-inf")
    inc_hit = incumbent.get("test_hit_rate") or float("-inf")
    cand_cov = candidate.get("test_decision_coverage") or 0.0
    inc_cov = incumbent.get("test_decision_coverage") or 0.0

    return {
        "auc_lift": cand_auc >= (inc_auc + 0.003),
        "realized_return_lift": cand_realized > inc_realized and cand_drawdown >= inc_drawdown,
        "hit_rate_lift_similar_coverage": cand_hit > inc_hit and abs(cand_cov - inc_cov) <= 0.05,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    base_args = [
        "--ohlcv",
        args.ohlcv,
        "--train-end",
        args.train_end,
        "--val-end",
        args.val_end,
        "--decision-cost",
        str(args.decision_cost),
        "--slippage",
        str(args.slippage),
        "--sweep-min-edge",
        "--sweep-start",
        str(args.sweep_start),
        "--sweep-end",
        str(args.sweep_end),
        "--sweep-step",
        str(args.sweep_step),
        "--min-coverage",
        str(args.min_coverage),
    ]

    experiment_specs: list[dict] = []

    for feature_set in ("baseline", "v2_numeric"):
        for calibration in ("sigmoid", "isotonic"):
            run_name = f"logistic__{feature_set}__{calibration}"
            experiment_specs.append(
                {
                    "name": run_name,
                    "script": repo_root / "src" / "train" / "train_baseline.py",
                    "args": [
                        "--feature-set",
                        feature_set,
                        "--calibration",
                        calibration,
                        "--out-dir",
                        str(runs_dir / run_name),
                    ],
                }
            )

    default_gbdt_params = {
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_iter": 400,
        "min_samples_leaf": 20,
    }
    for feature_set in ("baseline", "v2_numeric"):
        for calibration in ("sigmoid", "isotonic"):
            run_name = f"gbdt__{feature_set}__{calibration}__default"
            experiment_specs.append(
                {
                    "name": run_name,
                    "script": repo_root / "src" / "train" / "train_gbdt_baseline.py",
                    "args": [
                        "--feature-set",
                        feature_set,
                        "--calibration",
                        calibration,
                        "--learning-rate",
                        str(default_gbdt_params["learning_rate"]),
                        "--max-depth",
                        str(default_gbdt_params["max_depth"]),
                        "--max-iter",
                        str(default_gbdt_params["max_iter"]),
                        "--min-samples-leaf",
                        str(default_gbdt_params["min_samples_leaf"]),
                        "--out-dir",
                        str(runs_dir / run_name),
                    ],
                }
            )

    for calibration in ("sigmoid", "isotonic"):
        for max_depth in (4, 6, 8):
            for learning_rate in (0.03, 0.05):
                for min_samples_leaf in (20, 100):
                    for max_iter in (300, 500):
                        run_name = (
                            "gbdt__v2_numeric__"
                            f"{calibration}__d{max_depth}__lr{learning_rate}__leaf{min_samples_leaf}__iter{max_iter}"
                        )
                        experiment_specs.append(
                            {
                                "name": run_name,
                                "script": repo_root / "src" / "train" / "train_gbdt_baseline.py",
                                "args": [
                                    "--feature-set",
                                    "v2_numeric",
                                    "--calibration",
                                    calibration,
                                    "--learning-rate",
                                    str(learning_rate),
                                    "--max-depth",
                                    str(max_depth),
                                    "--max-iter",
                                    str(max_iter),
                                    "--min-samples-leaf",
                                    str(min_samples_leaf),
                                    "--out-dir",
                                    str(runs_dir / run_name),
                                ],
                            }
                        )

    for lookback in (48, 96):
        for hidden_dim in (64, 128):
            run_name = f"gru__v2_numeric__lb{lookback}__hd{hidden_dim}"
            experiment_specs.append(
                {
                    "name": run_name,
                    "script": repo_root / "src" / "train" / "train_sequence_model.py",
                    "args": [
                        "--feature-set",
                        "v2_numeric",
                        "--lookback",
                        str(lookback),
                        "--hidden-dim",
                        str(hidden_dim),
                        "--out-dir",
                        str(runs_dir / run_name),
                    ],
                }
            )

    rows = []
    model_args = []
    for spec in experiment_specs:
        command = [sys.executable, str(spec["script"]), *base_args, *spec["args"]]
        run_command(command, cwd=repo_root)

        metrics_path = runs_dir / spec["name"] / "metrics.json"
        model_args.append((spec["name"], metrics_path))
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        row = flatten_metrics(spec["name"], metrics)
        row["schema_warnings"] = schema_warnings(metrics)
        row["metrics_path"] = str(metrics_path)
        rows.append(row)

    comparison = {
        "n_models_loaded": len(rows),
        "rows": sorted(rows, key=lambda row: (row.get("test_auc") is None, -(row.get("test_auc") or -1))),
    }
    with (out_dir / "model_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    write_csv(comparison["rows"], out_dir / "model_comparison.csv")

    incumbent_name = "gbdt__baseline__isotonic__default"
    incumbent = next((row for row in rows if row["model_name"] == incumbent_name), None)
    if incumbent is None:
        raise ValueError(f"Unable to find incumbent run '{incumbent_name}'")

    ranked_rows = sorted(rows, key=lambda row: selection_key(row, args.min_coverage), reverse=True)
    candidate = ranked_rows[0]
    promotion_checks = beats_incumbent(candidate, incumbent)
    accepted = candidate["model_name"] != incumbent_name and any(promotion_checks.values())
    champion = candidate if accepted else incumbent

    report = {
        "selection_rules": {
            "validation_ranking": [
                "coverage >= min_coverage",
                "positive realized_total_pnl after cost/slippage",
                "higher val_auc",
                "less negative val_realized_max_drawdown_pct",
            ],
            "promotion_gate": {
                "auc_lift": "test_auc >= incumbent_test_auc + 0.003",
                "realized_return_lift": "higher test_realized_total_pnl and non-worse test_realized_max_drawdown_pct",
                "hit_rate_lift_similar_coverage": "higher test_hit_rate with absolute coverage delta <= 0.05",
            },
        },
        "incumbent": incumbent,
        "best_validation_candidate": candidate,
        "promotion_checks": promotion_checks,
        "accepted_new_champion": accepted,
        "champion": champion,
        "model_comparison_json": str(out_dir / "model_comparison.json"),
        "model_comparison_csv": str(out_dir / "model_comparison.csv"),
    }
    with (out_dir / "experiment_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved experiment report to {out_dir / 'experiment_report.json'}")


if __name__ == "__main__":
    main()
