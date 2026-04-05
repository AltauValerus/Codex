"""Quick connectivity preflight for exchange/API access."""

from __future__ import annotations

import argparse
import requests


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="https://api.coinbase.com/v2/currencies")
    p.add_argument("--timeout", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        r = requests.get(args.url, timeout=args.timeout)
        print({"reachable": True, "status_code": r.status_code, "url": args.url})
    except Exception as e:
        print({"reachable": False, "url": args.url, "error": str(e)})


if __name__ == "__main__":
    main()
