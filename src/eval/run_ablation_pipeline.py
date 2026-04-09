"""Run baseline/GBDT/GRU/CNN training and aggregate comparisons in one command.

This is an orchestration helper for quick ablation cycles.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--images-dir", default=None, help="Optional image dir for CNN run")
    p.add_argument("--windows", default=None, help="Optional window parquet for CNN run")
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--out-root", default="artifacts/ablation")
    p.add_argument("--manifest", default=None, help="Optional dataset manifest json")
    p.add_argument(
        "--gate-model",
        default="best",
        choices=["best", "logistic", "gbdt"],
        help="Model used for backtest + go/no-go gate. 'best' uses best_by_test_auc when supported.",
    )
    p.add_argument("--min-test-auc", type=float, default=0.55)
    p.add_argument("--min-test-balanced-accuracy", type=float, default=0.53)
    p.add_argument("--min-proxy-ev", type=float, default=0.0)
    p.add_argument("--min-trades", type=int, default=50)
    p.add_argument("--min-hit-rate", type=float, default=0.52)
    p.add_argument("--max-dd", type=float, default=-0.25)
    p.add_argument("--seq-epochs", type=int, default=6)
    p.add_argument("--cnn-epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def run(cmd: list[str]) -> int:
    print("\n$", " ".join(cmd))
    res = subprocess.run(cmd)
    return res.returncode


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "logistic": out_root / "baseline" / "metrics.json",
        "gbdt": out_root / "gbdt" / "metrics.json",
        "gru": out_root / "gru" / "metrics.json",
        "cnn": out_root / "cnn" / "metrics.json",
    }

    commands = [
        [
            "python",
            "-m",
            "src.train.train_baseline",
            "--ohlcv",
            args.ohlcv,
            "--train-end",
            args.train_end,
            "--val-end",
            args.val_end,
            "--out-dir",
            str(out_root / "baseline"),
        ],
        [
            "python",
            "-m",
            "src.train.train_gbdt_baseline",
            "--ohlcv",
            args.ohlcv,
            "--train-end",
            args.train_end,
            "--val-end",
            args.val_end,
            "--out-dir",
            str(out_root / "gbdt"),
        ],
        [
            "python",
            "-m",
            "src.train.train_sequence_model",
            "--ohlcv",
            args.ohlcv,
            "--train-end",
            args.train_end,
            "--val-end",
            args.val_end,
            "--epochs",
            str(args.seq_epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--out-dir",
            str(out_root / "gru"),
        ],
    ]

    for cmd in commands:
        code = run(cmd)
        if code != 0:
            raise SystemExit(f"Command failed with exit code {code}: {' '.join(cmd)}")

    cnn_enabled = bool(args.images_dir and args.windows and Path(args.images_dir).exists() and Path(args.windows).exists())
    if cnn_enabled:
        cnn_cmd = [
            "python",
            "-m",
            "src.train.train_image_cnn",
            "--images-dir",
            args.images_dir,
            "--windows",
            args.windows,
            "--epochs",
            str(args.cnn_epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--out-dir",
            str(out_root / "cnn"),
        ]
        code = run(cnn_cmd)
        if code != 0:
            print("[warn] CNN stage failed; continuing to comparison with available models.")
    else:
        print("[info] CNN stage skipped (images/windows not provided or not found).")

    compare_cmd = [
        "python",
        "-m",
        "src.eval.compare_models",
        "--model",
        f"logistic={paths['logistic']}",
        "--model",
        f"gbdt={paths['gbdt']}",
        "--model",
        f"gru={paths['gru']}",
        "--model",
        f"cnn={paths['cnn']}",
        "--out",
        str(out_root / "model_comparison.json"),
        "--out-csv",
        str(out_root / "model_comparison.csv"),
    ]
    code = run(compare_cmd)
    if code != 0:
        raise SystemExit(f"Comparison command failed with exit code {code}")

    # readiness + backtest + go/no-go chain
    readiness_cmd = [
        "python",
        "-m",
        "src.eval.readiness_report",
        "--comparison",
        str(out_root / "model_comparison.json"),
        "--min-test-auc",
        str(args.min_test_auc),
        "--min-test-balanced-accuracy",
        str(args.min_test_balanced_accuracy),
        "--min-proxy-ev",
        str(args.min_proxy_ev),
        "--out",
        str(out_root / "readiness_report.json"),
    ]
    code = run(readiness_cmd)
    if code != 0:
        raise SystemExit(f"Readiness command failed with exit code {code}")

    comparison = json.loads((out_root / "model_comparison.json").read_text(encoding="utf-8"))
    best_model_name = comparison.get("best_by_test_auc")
    requested_gate_model = best_model_name if args.gate_model == "best" else args.gate_model
    gate_model_to_artifact = {
        "logistic": out_root / "baseline" / "baseline_direction_model.joblib",
        "gbdt": out_root / "gbdt" / "gbdt_direction_model.joblib",
    }
    gate_model_name = requested_gate_model if requested_gate_model in gate_model_to_artifact else "logistic"
    if requested_gate_model not in gate_model_to_artifact:
        print(
            f"[warn] Requested gate model '{requested_gate_model}' is unsupported for backtest; "
            "falling back to 'logistic'."
        )

    backtest_cmd = [
        "python",
        "-m",
        "src.eval.backtest_from_model",
        "--model",
        str(gate_model_to_artifact[gate_model_name]),
        "--ohlcv",
        args.ohlcv,
        "--val-end",
        args.val_end,
        "--out",
        str(out_root / "backtest_baseline.json"),
    ]
    code = run(backtest_cmd)
    if code != 0:
        print("[warn] Backtest stage failed; skipping go/no-go gate.")
        print(f"\nDone. Check: {out_root / 'model_comparison.json'}")
        return

    go_no_go_cmd = [
        "python",
        "-m",
        "src.eval.go_no_go_gate",
        "--readiness",
        str(out_root / "readiness_report.json"),
        "--backtest",
        str(out_root / "backtest_baseline.json"),
        "--min-trades",
        str(args.min_trades),
        "--min-hit-rate",
        str(args.min_hit_rate),
        "--max-dd",
        str(args.max_dd),
        "--out",
        str(out_root / "go_no_go_report.json"),
    ]
    if args.manifest:
        go_no_go_cmd.extend(["--manifest", args.manifest])
    go_no_go_cmd.extend(["--gate-model-name", gate_model_name])

    code = run(go_no_go_cmd)
    if code != 0:
        raise SystemExit(f"Go/no-go command failed with exit code {code}")

    print(f"\nDone. Check: {out_root / 'go_no_go_report.json'}")


if __name__ == "__main__":
    main()
