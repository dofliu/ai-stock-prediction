"""The Makefile as a safety property, not a convenience.

`make clean` runs against a working tree that holds `data/journal/forecasts.csv`
- the one record in this project that cannot be regenerated, because every row
was written before its outcome existed. A `clean` that removes it destroys the
only score here that no amount of later tuning can fabricate, and an
uncommitted day is gone for good. It used to do exactly that (`rm -rf data`).

These tests pin the two properties that matter: what `clean` must never delete,
and that the local gate checks everything CI checks - while CI cannot run, the
Makefile is the only thing between a change and `main`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = REPO_ROOT / "Makefile"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")


def test_clean_keeps_the_forecast_journal_and_prices(tmp_path: Path) -> None:
    """The regression guard: `rm -rf data` must never come back.

    Runs the real recipe against a throwaway tree laid out like the repo's, so
    this fails if anyone widens `clean` back over the two protected paths.
    """
    shutil.copy(MAKEFILE, tmp_path / "Makefile")
    journal = tmp_path / "data" / "journal" / "forecasts.csv"
    prices = tmp_path / "data" / "prices" / "MU.csv"
    for path in (journal, prices):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("asof_date,symbol\n", encoding="utf-8")

    # The throwaway output `clean` is supposed to remove.
    disposable = [
        tmp_path / "data" / "synthetic.csv",
        tmp_path / "data" / "cache" / "MU.csv",
        tmp_path / "reports" / "backtest.md",
    ]
    for path in disposable:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("junk", encoding="utf-8")

    result = subprocess.run(
        ["make", "clean"], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr

    assert journal.exists(), "make clean deleted the forecast journal"
    assert journal.read_text(encoding="utf-8") == "asof_date,symbol\n"
    assert prices.exists(), "make clean deleted the price history"
    for path in disposable:
        assert not path.exists(), f"make clean left {path.name} behind"


def test_clean_succeeds_when_there_is_nothing_to_clean(tmp_path: Path) -> None:
    """A fresh checkout has no data/ at all; `clean` must not fail on it."""
    shutil.copy(MAKEFILE, tmp_path / "Makefile")

    result = subprocess.run(
        ["make", "clean"], cwd=tmp_path, capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr


def _planned_commands(target: str) -> str:
    """What `make <target>` would actually run, with variables expanded.

    Read through `make -n` rather than by matching the Makefile's text, so
    these assertions describe behaviour and survive the recipes being
    refactored into variables.
    """
    result = subprocess.run(
        ["make", "-n", target], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_lint_covers_the_same_paths_as_ci() -> None:
    """`scripts/` is the gap this closes: CI linted it, `make lint` did not.

    fetch_prices.py is the one path CI could never execute, so lint is the only
    checking it ever gets - and missing it locally means missing it entirely.
    """
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")
    assert "ruff check src tests scripts" in workflow
    assert "ruff format --check src tests scripts" in workflow

    planned = _planned_commands("lint")
    assert "ruff check src tests scripts" in planned
    assert "ruff format --check src tests scripts" in planned


def test_check_runs_every_gate_ci_runs() -> None:
    """`make check` is the promise that a clean local run means a clean CI."""
    planned = _planned_commands("check")

    assert "ruff check src tests scripts" in planned
    assert "ruff format --check src tests scripts" in planned
    assert "pytest -q" in planned
    assert "--doctest-modules src/ai_stock" in planned
    # The CLI smoke test, identified by the artefacts CI asserts are non-empty.
    for report in ("comparison.md", "simulation_ridge.md", "screen_ridge.md"):
        assert report in planned


def test_doctests_are_a_separate_gate_because_pytest_alone_skips_them() -> None:
    """`make test` runs none of them: testpaths is `tests`, doctests live in src."""
    assert "--doctest-modules src/ai_stock" in _planned_commands("doctest")
    assert "--doctest-modules" not in _planned_commands("test")


def test_smoke_does_not_write_into_the_working_tree() -> None:
    """CI smoke-tests on a throwaway runner; locally that would litter data/.

    A synthetic CSV written next to the real `data/prices/` is the start of
    exactly the confusion `data_freshness` exists to prevent.
    """
    planned = _planned_commands("smoke")

    assert "mktemp -d" in planned
    assert "data/prices" not in planned
    assert "--out reports" not in planned
