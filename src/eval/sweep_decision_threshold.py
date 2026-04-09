"""Select a decision threshold from validation predictions only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from pathlib import Path as _Path

import numpy as np
import pandas as pd

sys.path.append(str(_Path(__file__).resolve().parents[2]))

from src.eval.policy_utils import decision_report, sweep_min_edge


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True, help="Path to validation predictions parquet")
    p.add_argument("--decision-cost", type=float, default=0.02)
    p.add_argument("--slippage", type=float, default=0.0)
    p.add_argument("--min-coverage", type=float, default=0.05)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.08)
    p.add_argument("--step", type=float, default=0.01)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    predictions = pd.read_parquet(args.predictions)
    thresholds = np.arange(args.start, args.end + (args.step / 2.0), args.step)
    sweep = sweep_min_edge(
        predictions,
        thresholds=thresholds,
        decision_cost=args.decision_cost,
        slippage=args.slippage,
        min_coverage=args.min_coverage,
    )
    sweep["selected_report"] = decision_report(
        predictions,
        min_edge=sweep["selected_min_edge"],
        decision_cost=args.decision_cost,
        slippage=args.slippage,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(sweep, f, indent=2)

    print(json.dumps(sweep, indent=2))
    print(f"Saved threshold sweep to {out}")


if __name__ == "__main__":
    main()
