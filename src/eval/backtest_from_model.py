"""Backtest decision policy from a trained baseline-style classifier.

Loads a saved sklearn model and evaluates UP/DOWN/SKIP actions on a test period.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from src.features.common import build_feature_frame


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to .joblib model")
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--val-end", required=True, help="Test starts strictly after this UTC timestamp")
    p.add_argument("--min-edge", type=float, default=0.03)
    p.add_argument("--decision-cost", type=float, default=0.02)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--out", required=True, help="Output backtest json")
    return p.parse_args()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return build_feature_frame(df)


def max_drawdown_pct(equity_curve: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)

    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    df = df.sort_values("timestamp_open").reset_index(drop=True)

    X = build_features(df)
    y = (df["close"].shift(-1) > df["close"]).astype(int)

    data = X.copy()
    data["y"] = y
    data["timestamp_open"] = df["timestamp_open"]
    data = data.dropna().reset_index(drop=True)

    val_end = pd.Timestamp(args.val_end, tz="UTC")
    test = data[data["timestamp_open"] > val_end].copy()
    if len(test) == 0:
        raise SystemExit(
            "No rows in test period after --val-end. "
            "Pick an earlier --val-end or provide a longer OHLCV dataset."
        )

    feature_cols = [c for c in test.columns if c not in {"y", "timestamp_open"}]
    if not feature_cols:
        raise SystemExit("No feature columns available for backtest.")
    prob = model.predict_proba(test[feature_cols])[:, 1]

    edge = np.abs(prob - 0.5)
    action = np.where(edge >= args.min_edge, np.where(prob >= 0.5, 1, -1), 0)
    y_signed = np.where(test["y"].to_numpy() == 1, 1, -1)

    trade_mask = action != 0
    trade_pnl = np.zeros(len(action), dtype=float)
    trade_pnl[trade_mask] = np.where(action[trade_mask] == y_signed[trade_mask], 1.0, -1.0) - args.decision_cost - args.slippage

    equity = np.cumsum(trade_pnl)
    equity_curve = 100.0 + equity

    n_trades = int(np.sum(trade_mask))
    wins = int(np.sum((trade_pnl > 0) & trade_mask))
    losses = int(np.sum((trade_pnl < 0) & trade_mask))
    hit_rate = float(wins / n_trades) if n_trades else 0.0

    out_report = {
        "inputs": {
            "model": args.model,
            "ohlcv": args.ohlcv,
            "val_end": args.val_end,
            "min_edge": args.min_edge,
            "decision_cost": args.decision_cost,
            "slippage": args.slippage,
        },
        "summary": {
            "n_rows_test": int(len(test)),
            "n_trades": n_trades,
            "coverage": float(n_trades / max(len(test), 1)),
            "wins": wins,
            "losses": losses,
            "hit_rate": hit_rate,
            "avg_pnl_per_trade": float(np.mean(trade_pnl[trade_mask])) if n_trades else 0.0,
            "total_pnl": float(np.sum(trade_pnl)),
            "max_drawdown_pct": max_drawdown_pct(equity_curve) if len(equity_curve) else 0.0,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(out_report, f, indent=2)

    print(json.dumps(out_report, indent=2))
    print(f"Saved backtest report to {out}")


if __name__ == "__main__":
    main()
