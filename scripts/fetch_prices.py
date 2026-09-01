#!/usr/bin/env python3
"""Download the configured universe and write one CSV per symbol.

Run by the daily workflow, where network access exists. Individual tickers are
allowed to fail - a delisted or mistyped symbol should not cost you the day's
data for everything else - but a run in which *nothing* downloaded exits
non-zero, because that is an outage rather than a bad ticker.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ai_stock.data.loaders import save_csv
from ai_stock.pipeline import load_universe


def read_universe(path: Path) -> list[str]:
    """One ticker per line; ``#`` starts a comment."""
    if not path.exists():
        raise SystemExit(f"universe file not found: {path}")
    tickers = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ticker = line.split("#", 1)[0].strip()
        if ticker:
            tickers.append(ticker)
    if not tickers:
        raise SystemExit(f"{path} lists no tickers")
    return tickers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("config/universe.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/prices"))
    parser.add_argument("--period", default="15y", help="history length requested per ticker")
    args = parser.parse_args(argv)

    tickers = read_universe(args.universe)
    print(f"universe: {', '.join(tickers)}", flush=True)

    frames = load_universe(tickers=tickers, period=args.period, skip_errors=True)
    missing = [t for t in tickers if t not in frames]

    args.out.mkdir(parents=True, exist_ok=True)
    for symbol, frame in frames.items():
        path = save_csv(frame, args.out / f"{symbol.replace('/', '_')}.csv")
        print(
            f"  {symbol:12s} {len(frame):5d} bars  "
            f"{frame.index[0].date()} .. {frame.index[-1].date()}  -> {path}",
            flush=True,
        )

    if missing:
        print(f"failed: {', '.join(missing)}", file=sys.stderr, flush=True)
    if not frames:
        print("nothing downloaded - treating as an outage", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
