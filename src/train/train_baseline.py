"""Train a leakage-safe baseline for next-candle direction.

Usage:
python src/train/train_baseline.py \
  --ohlcv data/raw/ohlcv/coinbase_btcusd_5m.parquet \
  --train-end 2024-06-30T23:55:00Z \
  --val-end 2025-03-31T23:55:00Z \
  --decision-cost 0.02 \
  --min-edge 0.03 \
  --out-dir artifacts/baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--decision-cost", type=float, default=0.02)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat["ret_1"] = np.log(df["close"] / df["close"].shift(1))
    feat["ret_3"] = np.log(df["close"] / df["close"].shift(3))
    feat["ret_12"] = np.log(df["close"] / df["close"].shift(12))
    feat["range_hl"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    feat["body_oc"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    feat["vol_z_24"] = (
        (df["volume"] - df["volume"].rolling(24).mean())
        / (df["volume"].rolling(24).std().replace(0, np.nan))
    )
    return feat


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def decision_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    decision_cost: float,
    min_edge: float,
) -> dict[str, float]:
    edge = np.abs(y_prob - 0.5)
    take = edge >= min_edge

    if not np.any(take):
        return {
            "coverage": 0.0,
            "n_trades": 0,
            "hit_rate": 0.0,
            "avg_edge": 0.0,
            "proxy_ev_per_trade": -decision_cost,
        }

    y_sel = y_true[take]
    pred_sel = (y_prob[take] >= 0.5).astype(int)
    hit = (y_sel == pred_sel).astype(float)

    # Proxy EV for binary up/down decisions with cost deducted per action.
    # +1 for correct, -1 for wrong, then subtract fixed cost.
    ev = np.mean(np.where(hit > 0, 1.0, -1.0) - decision_cost)

    return {
        "coverage": float(np.mean(take)),
        "n_trades": int(np.sum(take)),
        "hit_rate": float(np.mean(hit)),
        "avg_edge": float(np.mean(edge[take])),
        "proxy_ev_per_trade": float(ev),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)

    X = build_features(df)
    y = (df["close"].shift(-1) > df["close"]).astype(int)

    dataset = X.copy()
    dataset["y"] = y
    dataset["timestamp_open"] = df["timestamp_open"]
    dataset = dataset.dropna().reset_index(drop=True)

    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_end = pd.Timestamp(args.val_end, tz="UTC")

    train_mask = dataset["timestamp_open"] <= train_end
    val_mask = (dataset["timestamp_open"] > train_end) & (dataset["timestamp_open"] <= val_end)
    test_mask = dataset["timestamp_open"] > val_end

    feature_cols = [c for c in dataset.columns if c not in {"y", "timestamp_open"}]
    X_train = dataset.loc[train_mask, feature_cols]
    y_train = dataset.loc[train_mask, "y"].to_numpy()
    X_val = dataset.loc[val_mask, feature_cols]
    y_val = dataset.loc[val_mask, "y"].to_numpy()
    X_test = dataset.loc[test_mask, feature_cols]
    y_test = dataset.loc[test_mask, "y"].to_numpy()

    base = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    model = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "objective": "direction_accuracy_with_cost_aware_filter",
        "decision_cost": args.decision_cost,
        "min_edge": args.min_edge,
        "val": {
            **compute_metrics(y_val, val_prob),
            "decision": decision_report(y_val, val_prob, args.decision_cost, args.min_edge),
        },
        "test": {
            **compute_metrics(y_test, test_prob),
            "decision": decision_report(y_test, test_prob, args.decision_cost, args.min_edge),
        },
        "counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "features": feature_cols,
    }

    joblib.dump(model, out_dir / "baseline_direction_model.joblib")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model + metrics to: {out_dir}")


if __name__ == "__main__":
    main()
