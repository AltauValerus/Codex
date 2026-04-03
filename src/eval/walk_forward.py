"""Walk-forward evaluation harness for leakage-resistant model checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--label-col", default="y")
    p.add_argument("--train-bars", type=int, default=400)
    p.add_argument("--test-bars", type=int, default=100)
    p.add_argument("--step-bars", type=int, default=100)
    p.add_argument("--out", required=True)
    return p.parse_args()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = np.log(df["close"] / df["close"].shift(1))
    feat["ret_3"] = np.log(df["close"] / df["close"].shift(3))
    feat["ret_12"] = np.log(df["close"] / df["close"].shift(12))
    feat["range_hl"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    feat["body_oc"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    return feat


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    args = parse_args()

    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)

    X = build_features(df)
    y = (df["close"].shift(-1) > df["close"]).astype(int)

    data = X.copy()
    data[args.label_col] = y
    data["timestamp_open"] = df["timestamp_open"]
    data = data.dropna().reset_index(drop=True)

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
        m = compute_metrics(te[args.label_col].to_numpy(), prob)
        fold_reports.append(
            {
                "train_start": str(tr["timestamp_open"].iloc[0]),
                "train_end": str(tr["timestamp_open"].iloc[-1]),
                "test_start": str(te["timestamp_open"].iloc[0]),
                "test_end": str(te["timestamp_open"].iloc[-1]),
                **m,
            }
        )

        start += args.step_bars

    agg = {
        "n_folds": len(fold_reports),
        "mean_auc": float(np.mean([f["auc"] for f in fold_reports])) if fold_reports else None,
        "mean_f1": float(np.mean([f["f1"] for f in fold_reports])) if fold_reports else None,
        "mean_balanced_accuracy": float(np.mean([f["balanced_accuracy"] for f in fold_reports])) if fold_reports else None,
    }

    report = {"aggregate": agg, "folds": fold_reports}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved walk-forward report to {out}")


if __name__ == "__main__":
    main()
