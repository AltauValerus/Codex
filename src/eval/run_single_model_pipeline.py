"""Minimal pipeline for single-model usage (no multi-model benchmarking)."""

from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--out-root", default="artifacts/single_model")
    p.add_argument("--decision-cost", type=float, default=0.02)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--min-hit-rate", type=float, default=0.52)
    p.add_argument("--max-dd", type=float, default=-0.25)
    return p.parse_args()


def run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd))
    return subprocess.run(cmd).returncode


def main() -> None:
    args = parse_args()

    baseline_dir = f"{args.out_root}/baseline"
    backtest_json = f"{args.out_root}/backtest_baseline.json"
    readiness_json = f"{args.out_root}/readiness_baseline.json"
    gate_json = f"{args.out_root}/go_no_go_baseline.json"

    steps = [
        [
            "python", "-m", "src.train.train_baseline",
            "--ohlcv", args.ohlcv,
            "--train-end", args.train_end,
            "--val-end", args.val_end,
            "--decision-cost", str(args.decision_cost),
            "--min-edge", str(args.min_edge),
            "--out-dir", baseline_dir,
        ],
        [
            "python", "-m", "src.eval.backtest_from_model",
            "--model", f"{baseline_dir}/baseline_direction_model.joblib",
            "--ohlcv", args.ohlcv,
            "--val-end", args.val_end,
            "--decision-cost", str(args.decision_cost),
            "--min-edge", str(args.min_edge),
            "--slippage", str(args.slippage),
            "--out", backtest_json,
        ],
        [
            "python", "-m", "src.eval.readiness_report",
            "--comparison", f"{args.out_root}/_single_comparison_tmp.json",
            "--out", readiness_json,
        ],
        [
            "python", "-m", "src.eval.go_no_go_gate",
            "--readiness", readiness_json,
            "--backtest", backtest_json,
            "--min-trades", str(args.min_trades),
            "--min-hit-rate", str(args.min_hit_rate),
            "--max-dd", str(args.max_dd),
            "--out", gate_json,
        ],
    ]

    prep_cmd = [
        "python",
        "-m",
        "src.eval.build_single_model_comparison",
        "--metrics",
        f"{baseline_dir}/metrics.json",
        "--model-name",
        "baseline",
        "--out",
        f"{args.out_root}/_single_comparison_tmp.json",
    ]

    if run(steps[0]) != 0:
        raise SystemExit("Baseline training failed")
    if run(prep_cmd) != 0:
        raise SystemExit("Failed to build temporary comparison file")
    for cmd in steps[1:]:
        if run(cmd) != 0:
            raise SystemExit(f"Step failed: {' '.join(cmd)}")

    print(f"\nDone. Single-model outputs available in: {args.out_root}")


if __name__ == "__main__":
    main()
