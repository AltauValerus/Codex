"""Build advanced labels for next-candle experiments.

Supports:
- fixed_horizon_ternary
- triple_barrier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--method", choices=["fixed_horizon_ternary", "triple_barrier"], required=True)
    p.add_argument("--horizon", type=int, default=1, help="Bars ahead for fixed horizon or timeout for triple barrier")
    p.add_argument("--flat-threshold", type=float, default=0.0005, help="Return threshold for ternary flat class")
    p.add_argument("--pt", type=float, default=0.0025, help="Profit-taking barrier for triple barrier")
    p.add_argument("--sl", type=float, default=0.0025, help="Stop-loss barrier for triple barrier")
    p.add_argument("--out", required=True)
    return p.parse_args()


def fixed_horizon_ternary(close: np.ndarray, horizon: int, flat_th: float) -> np.ndarray:
    out = np.full(len(close), np.nan)
    future = np.roll(close, -horizon)
    ret = np.log(future / close)

    out[ret > flat_th] = 1
    out[ret < -flat_th] = -1
    out[(ret >= -flat_th) & (ret <= flat_th)] = 0
    out[-horizon:] = np.nan
    return out


def triple_barrier(close: np.ndarray, horizon: int, pt: float, sl: float) -> np.ndarray:
    labels = np.full(len(close), np.nan)
    n = len(close)

    for i in range(n - horizon):
        p0 = close[i]
        up = p0 * (1 + pt)
        dn = p0 * (1 - sl)

        label = 0
        for j in range(i + 1, i + horizon + 1):
            px = close[j]
            if px >= up:
                label = 1
                break
            if px <= dn:
                label = -1
                break
        labels[i] = label

    return labels


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)
    close = df["close"].to_numpy(dtype=np.float64)

    if args.method == "fixed_horizon_ternary":
        y = fixed_horizon_ternary(close, args.horizon, args.flat_threshold)
    else:
        y = triple_barrier(close, args.horizon, args.pt, args.sl)

    out_df = pd.DataFrame(
        {
            "timestamp_open": df["timestamp_open"],
            "label": y,
            "method": args.method,
            "horizon": args.horizon,
        }
    ).dropna()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(out_df["label"].value_counts().to_dict())
    print(f"Saved labels to {out}")


if __name__ == "__main__":
    main()
