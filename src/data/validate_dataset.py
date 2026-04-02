"""Validate OHLCV and window-index integrity before training.

Usage:
python src/data/validate_dataset.py \
  --ohlcv data/raw/ohlcv/binance_btcusdt_5m.parquet \
  --windows data/interim/windows/window_index.parquet \
  --timeframe-minutes 5
"""

from __future__ import annotations

import argparse
import json

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--timeframe-minutes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ohlcv = pd.read_parquet(args.ohlcv)
    windows = pd.read_parquet(args.windows)

    ohlcv["timestamp_open"] = pd.to_datetime(ohlcv["timestamp_open"], utc=True)
    windows["timestamp_open"] = pd.to_datetime(windows["timestamp_open"], utc=True)

    ohlcv = ohlcv.sort_values("timestamp_open").reset_index(drop=True)
    delta = ohlcv["timestamp_open"].diff().dropna()
    expected = pd.Timedelta(minutes=args.timeframe_minutes)

    duplicate_ts = int(ohlcv["timestamp_open"].duplicated().sum())
    missing_gaps = int((delta > expected).sum())
    out_of_order = int((delta < expected).sum())

    split_order_ok = True
    split_rank = {"train": 0, "val": 1, "test": 2}
    split_seq = windows["split"].map(split_rank)
    if split_seq.isna().any() or not split_seq.is_monotonic_increasing:
        split_order_ok = False

    label_idx_ok = bool((windows["label_idx"] == windows["end_idx"] + 1).all())
    start_end_ok = bool((windows["start_idx"] <= windows["end_idx"]).all())

    class_balance = windows.groupby("split")["y_dir"].mean().to_dict()
    counts = windows["split"].value_counts().to_dict()

    report = {
        "ohlcv": {
            "rows": int(len(ohlcv)),
            "duplicate_timestamps": duplicate_ts,
            "missing_gaps": missing_gaps,
            "out_of_order_or_nonstandard_delta": out_of_order,
        },
        "windows": {
            "rows": int(len(windows)),
            "label_idx_equals_end_plus_one": label_idx_ok,
            "start_idx_le_end_idx": start_end_ok,
            "split_order_monotonic": split_order_ok,
            "counts_by_split": counts,
            "positive_rate_by_split": class_balance,
        },
    }

    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
