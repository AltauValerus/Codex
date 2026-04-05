"""Render deterministic candlestick images from window index."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def draw_candles(ax: plt.Axes, window: pd.DataFrame) -> None:
    x = range(len(window))
    up_color, down_color = "#16a34a", "#dc2626"

    for i, (_, row) in enumerate(window.iterrows()):
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = up_color if c >= o else down_color

        ax.plot([i, i], [l, h], color=color, linewidth=1)
        body_bottom = min(o, c)
        body_height = max(abs(c - o), 1e-8)
        rect = plt.Rectangle((i - 0.3, body_bottom), 0.6, body_height, color=color)
        ax.add_patch(rect)

    ax.set_xlim(-1, len(window))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    args = parse_args()
    ohlcv = pd.read_parquet(args.ohlcv)
    windows = pd.read_parquet(args.windows)

    base_out = Path(args.out_dir)
    for split in ["train", "val", "test"]:
        (base_out / split).mkdir(parents=True, exist_ok=True)

    dpi = args.dpi
    figsize = (args.width / dpi, args.height / dpi)

    if args.max_samples is not None:
        windows = windows.head(args.max_samples)

    for row in tqdm(windows.itertuples(index=False), total=len(windows), desc="Rendering"):
        start_idx = row.start_idx
        end_idx = row.end_idx
        split = row.split
        sample_id = row.sample_id

        window = ohlcv.iloc[start_idx : end_idx + 1]

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        draw_candles(ax, window)
        plt.tight_layout(pad=0)

        out_file = base_out / split / f"{sample_id}.png"
        fig.savefig(out_file, dpi=dpi, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Rendered {len(windows):,} images to {base_out}")


if __name__ == "__main__":
    main()
