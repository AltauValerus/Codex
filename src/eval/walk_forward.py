"""Walk-forward evaluation harness for leakage-resistant model checks."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.policy_utils import compute_classification_metrics
from src.features.common import build_direction_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--label-col", default="y")
    p.add_argument("--train-bars", type=int, default=400)
    p.add_argument("--test-bars", type=int, default=100)
    p.add_argument("--step-bars", type=int, default=100)
    p.add_argument("--feature-set", choices=["baseline", "v2_numeric"], default="baseline")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)

    data = build_direction_dataset(
        df,
        history_bars=24,
        horizon=1,
        timeframe_minutes=5,
        feature_set=args.feature_set,
    )
    if args.label_col != "y":
        data = data.rename(columns={"y": args.label_col})

    feature_cols = [c for c in data.columns if c not in {args.label_col, "timestamp_open"}]

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    n = len(data)
    start = 0
    fold_reports = []

    while True:
        train_start = start
        train_end = train_start + args.train_bars
        test_end = train_end + args.test_bars
        if test_end > n:
            break

        tr = data.iloc[train_start:train_end]
        te = data.iloc[train_end:test_end]

        model.fit(tr[feature_cols], tr[args.label_col])
        prob = model.predict_proba(te[feature_cols])[:, 1]
        metrics = compute_classification_metrics(te[args.label_col].to_numpy(), prob, include_brier=False)
        fold_reports.append(
            {
                "train_start": str(tr["timestamp_open"].iloc[0]),
                "train_end": str(tr["timestamp_open"].iloc[-1]),
                "test_start": str(te["timestamp_open"].iloc[0]),
                "test_end": str(te["timestamp_open"].iloc[-1]),
                **metrics,
            }
        )

        start += args.step_bars

    agg = {
        "n_folds": len(fold_reports),
        "mean_auc": float(np.nanmean([f["auc"] for f in fold_reports])) if fold_reports else None,
        "mean_f1": float(np.nanmean([f["f1"] for f in fold_reports])) if fold_reports else None,
        "mean_balanced_accuracy": float(np.nanmean([f["balanced_accuracy"] for f in fold_reports]))
        if fold_reports
        else None,
    }

    report = {
        "feature_set": args.feature_set,
        "label_col": args.label_col,
        "aggregate": agg,
        "folds": fold_reports,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved walk-forward report to {out}")


if __name__ == "__main__":
    main()
