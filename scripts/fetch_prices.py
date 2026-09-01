#!/usr/bin/env python3
"""Download the configured universe and write one CSV per symbol.

Run by the daily workflow, where network access exists.

Two behaviours matter for an unattended job:

* **Every failure is reported with its reason.** "failed: 2408.TW" tells you
  nothing you can act on at 22:00 on a Tuesday; the exception does.
* **Partial success saves the data and still fails the run.** The symbols that
  downloaded are written out so the day is not lost, and the exit code is
  non-zero so a universe quietly shrinking from four names to one is visible
  rather than green.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from ai_stock.data.loaders import load_yfinance, save_csv


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


def fetch(ticker: str, period: str, retries: int = 2) -> pd.DataFrame:
    """Download one ticker, retrying briefly on a transient failure."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return load_yfinance(ticker, period=period)
        except Exception as error:  # noqa: BLE001 - the reason is the payload
            last = error
            if attempt < retries:
                print(f"  {ticker}: attempt {attempt + 1} failed ({error}); retrying", flush=True)
    raise last  # type: ignore[misc]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", type=Path, default=Path("config/universe.txt"))
    parser.add_argument("--out", type=Path, default=Path("data/prices"))
    parser.add_argument("--period", default="15y", help="history length requested per ticker")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="exit 0 even when some tickers failed (default: any failure is an error)",
    )
    args = parser.parse_args(argv)

    tickers = read_universe(args.universe)
    print(f"universe: {', '.join(tickers)}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    failures: dict[str, str] = {}
    for ticker in tickers:
        try:
            frame = fetch(ticker, args.period)
        except Exception as error:  # noqa: BLE001 - reported, not raised
            failures[ticker] = f"{type(error).__name__}: {error}"
            traceback.print_exc(limit=3)
            continue
        path = save_csv(frame, args.out / f"{ticker.replace('/', '_')}.csv")
        print(
            f"  {ticker:12s} {len(frame):5d} bars  "
            f"{frame.index[0].date()} .. {frame.index[-1].date()}  -> {path}",
            flush=True,
        )

    succeeded = len(tickers) - len(failures)
    print(f"\n{succeeded}/{len(tickers)} symbols downloaded", flush=True)
    if failures:
        print("\nfailures:", file=sys.stderr)
        for ticker, reason in failures.items():
            print(f"  {ticker}: {reason}", file=sys.stderr)
        # The data written so far is already on disk; the caller commits it and
        # only then acts on this exit code.
        return 0 if args.allow_partial else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
