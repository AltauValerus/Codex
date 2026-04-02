"""Fetch historical OHLCV candles and store as parquet.

Example:
python src/data/fetch_ohlcv.py \
  --exchange binance --symbol BTC/USDT --timeframe 5m \
  --start 2021-01-01T00:00:00Z --end 2026-03-31T23:55:00Z \
  --out data/raw/ohlcv/binance_btcusdt_5m.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ccxt
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", required=True, help="ISO8601, e.g. 2021-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="ISO8601, e.g. 2026-03-31T23:55:00Z")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def fetch_all(
    exchange_name: str,
    symbol: str,
    timeframe: str,
    start_iso: str,
    end_iso: str,
    limit: int,
) -> pd.DataFrame:
    exchange_class = getattr(ccxt, exchange_name)
    exchange = exchange_class({"enableRateLimit": True})

    since = exchange.parse8601(start_iso)
    end_ms = exchange.parse8601(end_iso)

    rows: list[list[float]] = []

    pbar = tqdm(desc="Fetching OHLCV", unit="batch")
    while since < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        pbar.update(1)

        if not batch:
            break

        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since:
            break

        tf_ms = exchange.parse_timeframe(timeframe) * 1000
        since = last_ts + tf_ms

    pbar.close()

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"] <= end_ms].copy()
    df["timestamp_open"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df[["timestamp_open", "open", "high", "low", "close", "volume"]]


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = fetch_all(
        exchange_name=args.exchange,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_iso=args.start,
        end_iso=args.end,
        limit=args.limit,
    )

    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
