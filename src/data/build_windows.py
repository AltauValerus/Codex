"""Build window index and labels for next-candle prediction."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from src.features.common import compute_contiguous_sample_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--window-size", type=int, default=96)
    parser.add_argument("--train-end", required=True, help="ISO8601 UTC")
    parser.add_argument("--val-end", required=True, help="ISO8601 UTC")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def assign_split(ts: pd.Timestamp, train_end: pd.Timestamp, val_end: pd.Timestamp) -> str:
    if ts <= train_end:
        return "train"
    if ts <= val_end:
        return "val"
    return "test"


def main() -> None:
    args = parse_args()
    df = pd.read_parquet(args.ohlcv)
    df["timestamp_open"] = pd.to_datetime(df["timestamp_open"], utc=True)

    close = df["close"].to_numpy(dtype=np.float64)
    ts = df["timestamp_open"].to_numpy()

    indices = np.arange(args.window_size - 1, len(df) - 1)
    current_close = close[indices]
    next_close = close[indices + 1]

    y_dir = (next_close > current_close).astype(np.int8)
    y_ret = np.log(next_close / current_close)

    train_end = pd.Timestamp(args.train_end, tz="UTC")
    val_end = pd.Timestamp(args.val_end, tz="UTC")
    contiguous_mask = compute_contiguous_sample_mask(
        df["timestamp_open"],
        history_bars=args.window_size,
        horizon=1,
        timeframe_minutes=5,
    ).to_numpy()

    rows = []
    skipped = 0
    for idx in indices:
        if not contiguous_mask[idx]:
            skipped += 1
            continue
        sample_pos = idx - (args.window_size - 1)
        t = pd.Timestamp(ts[idx])
        split = assign_split(t, train_end, val_end)
        rows.append(
            {
                "sample_id": len(rows),
                "end_idx": int(idx),
                "start_idx": int(idx - args.window_size + 1),
                "label_idx": int(idx + 1),
                "timestamp_open": t,
                "y_dir": int(y_dir[sample_pos]),
                "y_ret": float(y_ret[sample_pos]),
                "split": split,
            }
        )

    out_df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out, index=False)
    print(out_df["split"].value_counts(dropna=False).to_dict())
    print(f"Skipped {skipped:,} windows that crossed missing bars")
    print(f"Saved {len(out_df):,} windows to {out}")


if __name__ == "__main__":
    main()
