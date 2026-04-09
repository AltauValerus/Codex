"""Fetch historical OHLCV candles and store as parquet.

Example:
python -m src.data.fetch_ohlcv \
  --exchange binance --symbol BTC/USDT --timeframe 5m \
  --start 2021-01-01T00:00:00Z --end 2026-03-31T23:55:00Z \
  --out data/raw/ohlcv/binance_btcusdt_5m.parquet
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ccxt
import pandas as pd
import requests
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["ccxt", "alphavantage_rapidapi"], default="ccxt")
    parser.add_argument("--exchange", default="binance")
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--start", required=True, help="ISO8601, e.g. 2021-01-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="ISO8601, e.g. 2026-03-31T23:55:00Z")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--rapidapi-key", default=None, help="Defaults to RAPIDAPI_KEY env var")
    parser.add_argument("--rapidapi-host", default="alpha-vantage.p.rapidapi.com")
    parser.add_argument("--market", default="USD", help="Quote currency for Alpha Vantage crypto endpoint")
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


def fetch_alphavantage_rapidapi(
    symbol: str,
    timeframe: str,
    start_iso: str,
    end_iso: str,
    rapidapi_key: str,
    rapidapi_host: str,
    market: str,
) -> pd.DataFrame:
    crypto_symbol = symbol.split("/")[0].replace("-", "").upper()
    url = f"https://{rapidapi_host}/query"
    params = {
        "function": "CRYPTO_INTRADAY",
        "symbol": crypto_symbol,
        "market": market,
        "interval": timeframe,
        "outputsize": "full",
    }
    headers = {
        "X-RapidAPI-Key": rapidapi_key,
        "X-RapidAPI-Host": rapidapi_host,
    }
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()

    series_key = next((k for k in payload.keys() if k.startswith("Time Series Crypto")), None)
    if not series_key:
        raise RuntimeError(
            f"Alpha Vantage payload did not include a time-series key. "
            f"Response keys: {list(payload.keys())[:10]}"
        )

    rows: list[dict] = []
    for ts, values in payload[series_key].items():
        rows.append(
            {
                "timestamp_open": pd.Timestamp(ts, tz="UTC"),
                "open": float(values.get("1. open")),
                "high": float(values.get("2. high")),
                "low": float(values.get("3. low")),
                "close": float(values.get("4. close")),
                "volume": float(values.get("5. volume")),
            }
        )

    df = pd.DataFrame(rows).sort_values("timestamp_open").reset_index(drop=True)
    start_ts = pd.Timestamp(start_iso, tz="UTC")
    end_ts = pd.Timestamp(end_iso, tz="UTC")
    df = df[(df["timestamp_open"] >= start_ts) & (df["timestamp_open"] <= end_ts)].copy()
    return df[["timestamp_open", "open", "high", "low", "close", "volume"]]


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.provider == "ccxt":
        df = fetch_all(
            exchange_name=args.exchange,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_iso=args.start,
            end_iso=args.end,
            limit=args.limit,
        )
    else:
        rapidapi_key = args.rapidapi_key or os.getenv("RAPIDAPI_KEY")
        if not rapidapi_key:
            raise SystemExit(
                "Missing RapidAPI key. Pass --rapidapi-key or set RAPIDAPI_KEY in environment."
            )
        df = fetch_alphavantage_rapidapi(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_iso=args.start,
            end_iso=args.end,
            rapidapi_key=rapidapi_key,
            rapidapi_host=args.rapidapi_host,
            market=args.market,
        )

    df.to_parquet(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")


if __name__ == "__main__":
    main()
