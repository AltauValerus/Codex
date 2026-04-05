"""Train a gradient-boosted baseline for next-candle direction."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.features.common import build_feature_frame
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--decision-cost", type=float, default=0.02)
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_feature_frame(df)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }


def decision_report(y_true: np.ndarray, y_prob: np.ndarray, decision_cost: float, min_edge: float) -> dict[str, float]:
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

    base = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=400,
        min_samples_leaf=20,
        random_state=42,
    )
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": "hist_gradient_boosting",
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

    joblib.dump(model, out_dir / "gbdt_direction_model.joblib")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model + metrics to: {out_dir}")


if __name__ == "__main__":
    main()
