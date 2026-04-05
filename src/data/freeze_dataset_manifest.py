"""Freeze dataset manifest with hashes and metadata for reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Dataset name, e.g. dataset_v1")
    p.add_argument("--timeframe", required=True, help="e.g. 5m")
    p.add_argument("--source", required=True, help="e.g. coinbase BTC/USD")
    p.add_argument("--train-end", required=True)
    p.add_argument("--val-end", required=True)
    p.add_argument("--ohlcv", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--images-dir", default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path) -> dict:
    info = {
        "path": str(path),
        "exists": path.exists(),
        "sha256": sha256_file(path) if path.exists() else None,
        "bytes": path.stat().st_size if path.exists() else None,
    }
    if path.exists() and path.suffix == ".parquet":
        try:
            info["rows"] = int(len(pd.read_parquet(path)))
        except Exception:
            info["rows"] = None
    return info


def image_counts(images_dir: Path) -> dict:
    out = {}
    for split in ["train", "val", "test"]:
        split_dir = images_dir / split
        out[split] = len(list(split_dir.glob("*.png"))) if split_dir.exists() else 0
    out["total"] = sum(out.values())
    return out


def main() -> None:
    args = parse_args()

    ohlcv = Path(args.ohlcv)
    windows = Path(args.windows)

    manifest = {
        "dataset_name": args.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "timeframe": args.timeframe,
        "splits": {
            "train_end": args.train_end,
            "val_end": args.val_end,
        },
        "files": {
            "ohlcv": file_info(ohlcv),
            "windows": file_info(windows),
        },
    }

    if args.images_dir:
        img_dir = Path(args.images_dir)
        manifest["files"]["images_dir"] = {
            "path": str(img_dir),
            "exists": img_dir.exists(),
            "counts": image_counts(img_dir) if img_dir.exists() else {"train": 0, "val": 0, "test": 0, "total": 0},
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps(manifest, indent=2))
    print(f"Saved dataset manifest to {out}")


if __name__ == "__main__":
    main()
