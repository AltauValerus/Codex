"""Train a leakage-safe logistic baseline for next-candle direction."""

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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.policy_utils import (
    build_prediction_frame,
    compute_classification_metrics,
    decision_report,
    sweep_min_edge,
)
from src.features.common import (
    PREDICTION_CONTEXT_COLUMNS,
    build_direction_dataset,
    build_prediction_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--decision-cost", type=float, default=0.02)
    parser.add_argument("--min-edge", type=float, default=0.03)
    parser.add_argument("--slippage", type=float, default=0.0)
    parser.add_argument("--feature-set", choices=["baseline", "v2_numeric"], default="baseline")
    parser.add_argument("--calibration", choices=["sigmoid", "isotonic"], default="sigmoid")
    parser.add_argument("--sweep-min-edge", action="store_true")
    parser.add_argument("--sweep-start", type=float, default=0.0)
    parser.add_argument("--sweep-end", type=float, default=0.08)
    parser.add_argument("--sweep-step", type=float, default=0.01)
    parser.add_argument("--min-coverage", type=float, default=0.05)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)

    dataset = build_direction_dataset(
        df,
        history_bars=24,
        horizon=1,
        timeframe_minutes=5,
        feature_set=args.feature_set,
    )
    context = build_prediction_context(df, horizon=1)
    dataset = dataset.merge(context, on="timestamp_open", how="left")
    dataset = dataset.dropna(subset=PREDICTION_CONTEXT_COLUMNS).reset_index(drop=True)

    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_end = pd.Timestamp(args.val_end, tz="UTC")

    train_mask = dataset["timestamp_open"] <= train_end
    val_mask = (dataset["timestamp_open"] > train_end) & (dataset["timestamp_open"] <= val_end)
    test_mask = dataset["timestamp_open"] > val_end

    feature_cols = [
        c for c in dataset.columns if c not in {"y", "timestamp_open", *PREDICTION_CONTEXT_COLUMNS}
    ]
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
    model = CalibratedClassifierCV(base, method=args.calibration, cv=3)
    model.fit(X_train, y_train)

    train_prob = model.predict_proba(X_train)[:, 1]
    val_prob = model.predict_proba(X_val)[:, 1]
    test_prob = model.predict_proba(X_test)[:, 1]

    train_metrics = compute_classification_metrics(y_train, train_prob)
    val_metrics = compute_classification_metrics(y_val, val_prob)
    test_metrics = compute_classification_metrics(y_test, test_prob)

    prediction_frames = {
        "train": build_prediction_frame(
            dataset.loc[train_mask].reset_index(drop=True),
            train_prob,
            split="train",
            model_name="logistic",
            feature_set=args.feature_set,
            calibration=args.calibration,
        ),
        "val": build_prediction_frame(
            dataset.loc[val_mask].reset_index(drop=True),
            val_prob,
            split="val",
            model_name="logistic",
            feature_set=args.feature_set,
            calibration=args.calibration,
        ),
        "test": build_prediction_frame(
            dataset.loc[test_mask].reset_index(drop=True),
            test_prob,
            split="test",
            model_name="logistic",
            feature_set=args.feature_set,
            calibration=args.calibration,
        ),
    }

    prediction_files: dict[str, str] = {}
    for split_name, pred_df in prediction_frames.items():
        pred_path = out_dir / f"predictions_{split_name}.parquet"
        pred_df.to_parquet(pred_path, index=False)
        prediction_files[split_name] = str(pred_path)

    threshold_sweep = None
    selected_min_edge = float(args.min_edge)
    if args.sweep_min_edge:
        thresholds = np.arange(args.sweep_start, args.sweep_end + (args.sweep_step / 2.0), args.sweep_step)
        threshold_sweep = sweep_min_edge(
            prediction_frames["val"],
            thresholds=thresholds,
            decision_cost=args.decision_cost,
            slippage=args.slippage,
            min_coverage=args.min_coverage,
            auc_tiebreaker=val_metrics["auc"],
        )
        selected_min_edge = float(threshold_sweep["selected_min_edge"])

    metrics = {
        "model_name": "logistic",
        "objective": "direction_accuracy_with_cost_aware_filter",
        "feature_set": args.feature_set,
        "calibration": args.calibration,
        "decision_cost": args.decision_cost,
        "slippage": args.slippage,
        "configured_min_edge": args.min_edge,
        "selected_min_edge": selected_min_edge,
        "threshold_selection": threshold_sweep,
        "train": {
            **train_metrics,
            "decision": decision_report(
                prediction_frames["train"],
                min_edge=selected_min_edge,
                decision_cost=args.decision_cost,
                slippage=args.slippage,
            ),
        },
        "val": {
            **val_metrics,
            "decision": decision_report(
                prediction_frames["val"],
                min_edge=selected_min_edge,
                decision_cost=args.decision_cost,
                slippage=args.slippage,
            ),
        },
        "test": {
            **test_metrics,
            "decision": decision_report(
                prediction_frames["test"],
                min_edge=selected_min_edge,
                decision_cost=args.decision_cost,
                slippage=args.slippage,
            ),
        },
        "counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "features": feature_cols,
        "prediction_files": prediction_files,
    }

    joblib.dump(model, out_dir / "baseline_direction_model.joblib")
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved model + metrics to: {out_dir}")


if __name__ == "__main__":
    main()
