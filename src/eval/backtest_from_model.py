"""Backtest decision policy from a trained baseline-style classifier."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.eval.policy_utils import build_prediction_frame, compute_classification_metrics, decision_report
from src.features.common import (
    PREDICTION_CONTEXT_COLUMNS,
    build_direction_dataset,
    build_prediction_context,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to .joblib model")
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--val-end", required=True, help="Test starts strictly after this UTC timestamp")
    p.add_argument("--feature-set", choices=["baseline", "v2_numeric"], default="baseline")
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--decision-cost", type=float, default=0.02)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--out", required=True, help="Output backtest json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)

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
    context = build_prediction_context(df, horizon=1)
    data = data.merge(context, on="timestamp_open", how="left")
    data = data.dropna(subset=PREDICTION_CONTEXT_COLUMNS).reset_index(drop=True)

    val_end = pd.Timestamp(args.val_end, tz="UTC")
    test = data[data["timestamp_open"] > val_end].copy()

    feature_cols = [c for c in test.columns if c not in {"y", "timestamp_open", *PREDICTION_CONTEXT_COLUMNS}]
    prob = model.predict_proba(test[feature_cols])[:, 1]
    prediction_df = build_prediction_frame(
        test.reset_index(drop=True),
        prob,
        split="test",
        model_name="loaded_model",
        feature_set=args.feature_set,
        calibration=None,
    )

    out_report = {
        "inputs": {
            "model": args.model,
            "ohlcv": args.ohlcv,
            "val_end": args.val_end,
            "feature_set": args.feature_set,
            "min_edge": args.min_edge,
            "decision_cost": args.decision_cost,
            "slippage": args.slippage,
        },
        "classification": compute_classification_metrics(
            prediction_df["y_true"].to_numpy(),
            prediction_df["y_prob"].to_numpy(),
        ),
        "summary": decision_report(
            prediction_df,
            min_edge=args.min_edge,
            decision_cost=args.decision_cost,
            slippage=args.slippage,
        ),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)

    print(json.dumps(out_report, indent=2))
    print(f"Saved backtest report to {out}")


if __name__ == "__main__":
    main()
