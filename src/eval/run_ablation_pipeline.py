"""Run baseline/GBDT/GRU/CNN training and aggregate comparisons in one command.

This is an orchestration helper for quick ablation cycles.
"""

from __future__ import annotations

import argparse
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
            "src/train/train_baseline.py",
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
            "src/train/train_gbdt_baseline.py",
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
            "src/train/train_sequence_model.py",
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
            "src/train/train_image_cnn.py",
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
        "src/eval/compare_models.py",
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

    print(f"\nDone. Check: {out_root / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
