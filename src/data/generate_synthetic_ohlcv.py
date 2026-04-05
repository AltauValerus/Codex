"""Generate synthetic OHLCV for offline smoke tests.

Useful when external exchange APIs are unreachable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True)
    p.add_argument("--periods", type=int, default=2000)
    p.add_argument("--timeframe-minutes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start-price", type=float, default=50000.0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ts = pd.date_range(args.start, periods=args.periods, freq=f"{args.timeframe_minutes}min", tz="UTC")

    rets = rng.normal(loc=0.0, scale=0.0015, size=args.periods)
    close = args.start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate(([args.start_price], close[:-1]))

    wick = np.abs(rng.normal(0, 0.0008, size=args.periods))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)

    volume = rng.lognormal(mean=6.0, sigma=0.35, size=args.periods)

    df = pd.DataFrame(
        {
            "timestamp_open": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved synthetic OHLCV: {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
