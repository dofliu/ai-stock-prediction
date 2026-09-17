#!/usr/bin/env python3
"""Run the daily fetch-record-score-commit loop on a developer machine.

`.github/workflows/daily-prices.yml` does this every weekday on a runner. While
the Actions quota is exhausted it does not run at all, and the forecast journal
stops accumulating - which matters more than it sounds, because a journal that
is not growing is not merely idle, it is dead: `hit_rate_z` needs independent
horizons, and those only arrive one trading day at a time.

This script is that workflow, minus the runner. It keeps the one ordering the
workflow got right:

    fetch -> record and score -> commit -> *then* report a download failure

A partial outage must not cost the bars that did arrive, so the data is
committed before the run is allowed to go red.

It also adds the guards a workflow never needed and a laptop does. A runner
starts from a clean checkout of `main`; a working tree does not, and the
failure modes are unforgiving:

* **Committing from the wrong branch.** Refuses to run unless the checkout is
  on `--branch` (default `main`).
* **Sweeping unrelated staged work into a data commit.** `git commit` commits
  the whole index, not the paths you just added. This uses ``--only``, so
  whatever else you had staged stays staged and unpushed.
* **A rewritten journal.** The single thing this repository cannot recover
  from. Every row was written before its outcome existed, so a row that
  changes is a row that was recomputed with hindsight, and re-recording a lost
  day would fit the model on data the original never saw.
  :func:`journal_is_append_only` compares the file against the committed
  version and aborts *before* the commit if any existing row moved. Nothing in
  this script ever rewrites the journal; the check is there for the day
  something else does.

Usage::

    python3 scripts/daily_update.py              # the real thing
    python3 scripts/daily_update.py --dry-run    # fetch and score, commit nothing
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

JOURNAL = Path("data/journal/forecasts.csv")
PRICES = Path("data/prices")
COMMITTED_PATHS = (str(PRICES), str(JOURNAL.parent))


# --------------------------------------------------------------------------- #
# The safety property: a journal only ever grows
# --------------------------------------------------------------------------- #
def journal_is_append_only(previous: str, current: str) -> bool:
    """Whether ``current`` extends ``previous`` without altering a single row.

    The forecast journal is append-only by construction, and that is the whole
    basis for trusting it: a row written before its outcome existed cannot be
    tuned afterwards, but only if it is still the row that was written. An
    edited row is indistinguishable from an honest one once it is on disk, so
    the check has to happen before the commit rather than in review.

    A shorter file, a changed header, or any changed line is a rewrite. An
    empty or missing previous version is not - that is the first run.

    >>> journal_is_append_only("h\\na\\n", "h\\na\\nb\\n")
    True
    >>> journal_is_append_only("h\\na\\n", "h\\nEDITED\\nb\\n")
    False
    >>> journal_is_append_only("h\\na\\nb\\n", "h\\na\\n")
    False
    >>> journal_is_append_only("", "h\\na\\n")
    True
    """
    old = previous.splitlines()
    new = current.splitlines()
    if not old:
        return True
    if len(new) < len(old):
        return False
    return new[: len(old)] == old


def rows_added(previous: str, current: str) -> int:
    """How many lines ``current`` gained over ``previous``.

    >>> rows_added("h\\na\\n", "h\\na\\nb\\nc\\n")
    2
    """
    return len(current.splitlines()) - len(previous.splitlines())


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #
def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def current_branch() -> str:
    return git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def committed_journal() -> str:
    """The journal as `HEAD` has it, or empty if it is not committed yet."""
    result = git("show", f"HEAD:{JOURNAL.as_posix()}", check=False)
    return result.stdout if result.returncode == 0 else ""


def has_staged_changes() -> bool:
    return git("diff", "--cached", "--quiet", check=False).returncode != 0


def paths_are_dirty() -> bool:
    result = git("status", "--porcelain", "--", *COMMITTED_PATHS)
    return bool(result.stdout.strip())


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
def run(label: str, argv: list[str]) -> int:
    """Run a step, streaming its output, and return its exit code."""
    print(f"\n=== {label} ===", flush=True)
    return subprocess.run(argv, check=False).returncode


def preflight(branch: str) -> None:
    """Refuse to start when the tree is not somewhere it is safe to commit from."""
    if git("rev-parse", "--is-inside-work-tree", check=False).returncode != 0:
        raise SystemExit("not inside a git work tree")

    actual = current_branch()
    if actual != branch:
        raise SystemExit(
            f"on branch {actual!r}, expected {branch!r}.\n"
            f"This script commits and pushes the day's data; running it from a "
            f"feature branch would put the record somewhere it will be lost.\n"
            f"Either `git checkout {branch}` or pass --branch {actual}."
        )

    if has_staged_changes():
        print(
            "note: you have staged changes. They will be left staged and "
            "unpushed - only the data paths are committed.",
            file=sys.stderr,
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="random_forest")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--period", default="15y")
    parser.add_argument("--cost-bps", type=float, default=2.0)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--branch", default="main", help="branch this may commit to")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and score, then stop: commit nothing and push nothing",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="score the bars already on disk")
    args = parser.parse_args(argv)

    preflight(args.branch)
    before = committed_journal()

    # Recorded but not raised. The bars that arrived must reach the commit even
    # if some symbol failed, or a partial outage costs the whole day.
    fetch_status = 0
    if not args.skip_fetch:
        fetch_status = run(
            "fetch prices",
            [sys.executable, "scripts/fetch_prices.py", "--period", args.period],
        )
        if fetch_status != 0:
            print("one or more symbols failed to download (reported after the commit)", flush=True)

    journal_status = run(
        "record and score",
        [
            sys.executable,
            "-m",
            "ai_stock",
            "journal",
            "--data",
            str(PRICES),
            "--journal",
            str(JOURNAL),
            "--model",
            args.model,
            "--horizon",
            str(args.horizon),
            "--cost-bps",
            str(args.cost_bps),
            "--slippage-bps",
            str(args.slippage_bps),
            "--no-compare",
            "--out",
            str(args.out),
        ],
    )
    if journal_status != 0:
        print("\njournal step failed; nothing committed", file=sys.stderr)
        return journal_status

    after = JOURNAL.read_text(encoding="utf-8") if JOURNAL.exists() else ""
    if not journal_is_append_only(before, after):
        print(
            "\nREFUSING TO COMMIT: the journal is not an extension of the "
            "committed version - an existing row changed.\n"
            "Every row was written before its outcome existed; a row that moved "
            "was recomputed with hindsight, and that is the one thing this "
            "record cannot survive.\n"
            f"Inspect with: git diff -- {JOURNAL}",
            file=sys.stderr,
        )
        return 2
    print(f"\njournal: {rows_added(before, after)} row(s) added, none altered", flush=True)

    if args.dry_run:
        print("--dry-run: nothing committed or pushed", flush=True)
        return fetch_status

    if not paths_are_dirty():
        print("no new bars or forecasts - nothing to commit", flush=True)
        return fetch_status

    # `datetime.UTC` is 3.11+; this package supports 3.10.
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # --only commits these paths alone, whatever else sits in the index.
    commit = git(
        "commit",
        "--only",
        *COMMITTED_PATHS,
        "-m",
        f"Update prices and forecast journal ({stamp}) [skip ci]",
        check=False,
    )
    print(commit.stdout or commit.stderr, flush=True)
    if commit.returncode != 0:
        print("commit failed; nothing pushed", file=sys.stderr)
        return commit.returncode

    # Our commit is one append on top; rebasing it onto a moved `main` is safe
    # and keeps the data history linear.
    if git("pull", "--rebase", "origin", args.branch, check=False).returncode != 0:
        print(
            "pull --rebase failed - the commit is local and safe. Resolve, then "
            "`git push origin " + args.branch + "`.",
            file=sys.stderr,
        )
        return 1

    if git("push", "origin", args.branch, check=False).returncode != 0:
        print("push failed - the commit is local. Retry `git push`.", file=sys.stderr)
        return 1

    print(f"pushed to {args.branch}", flush=True)
    return fetch_status


if __name__ == "__main__":
    raise SystemExit(main())
