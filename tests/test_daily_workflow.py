"""The daily workflow as an ordering property, not a convenience.

`.github/workflows/daily-prices.yml` is the only thing that feeds the forecast
journal, and it gets one bargain right that is easy to undo by accident: the
day's bars are committed *before* the run is allowed to go red. A partial
outage, or a feed that has stopped, must not also cost the data that did
arrive - a lost day cannot be backfilled, because a forecast written after the
outcome exists is not a forecast.

Both alarms in that file therefore live at the end, reading a step's `outcome`
rather than failing where the problem was found. These tests pin that layout,
so moving `--fail-if-stale` up into the job - the obvious-looking tidy-up -
fails here instead of silently costing a day the next time the feed stops.

Read as text rather than parsed as YAML: PyYAML is not a declared dependency of
this project, and the property under test is about the order steps appear in,
which the text shows directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "daily-prices.yml"


@pytest.fixture(scope="module")
def workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _position(text: str, needle: str) -> int:
    index = text.find(needle)
    assert index != -1, f"{needle!r} is no longer in daily-prices.yml"
    return index


def test_the_journal_step_asks_for_the_stale_feed_alarm(workflow: str) -> None:
    assert "--fail-if-stale" in workflow


def test_the_stale_alarm_is_wider_than_the_bare_flag_default(workflow: str) -> None:
    """A build alarm that cries wolf every Lunar New Year is one nobody reads.

    The bare flag defaults to four calendar days, which suits a person glancing
    at output, where a needless glance costs a glance. This exit fails a build,
    and the Taiwan market shuts for up to nine calendar days in February, so
    the workflow must pass its own wider figure rather than accepting that
    default.
    """
    from ai_stock.journal import STALE_AFTER_DAYS

    # The argument itself, not the comment above it that names the same flag.
    line = next(
        ln for ln in workflow.splitlines() if ln.strip().startswith("--fail-if-stale")
    ).strip()
    threshold = int(line.removeprefix("--fail-if-stale").rstrip("\\").strip())
    assert threshold > STALE_AFTER_DAYS
    assert threshold >= 10, "must clear a Lunar New Year closure"


def test_the_stale_alarm_fires_after_the_data_is_committed(workflow: str) -> None:
    """The property this file exists for. Breaking it costs a trading day."""
    journal_step = _position(workflow, "Record today's forecasts and score the matured ones")
    commit_step = _position(workflow, "Commit the day's data")
    alarm_step = _position(workflow, "Fail if the price feed has stopped")

    assert journal_step < commit_step < alarm_step


def test_the_journal_step_cannot_abort_the_commit(workflow: str) -> None:
    """`continue-on-error` is what lets the alarm sit after the commit at all.

    Without it a stale-feed exit ends the job where it happened, the commit step
    never runs, and the outage costs the bars that did download - the exact
    failure the ordering above is designed to prevent.
    """
    steps = workflow.split("      - name: ")
    journal = next(s for s in steps if s.startswith("Record today's forecasts"))
    assert "continue-on-error: true" in journal
    assert "id: journal" in journal


def test_both_alarms_read_an_outcome_rather_than_failing_in_place(workflow: str) -> None:
    assert "if: steps.fetch.outcome == 'failure'" in workflow
    assert "if: steps.journal.outcome == 'failure'" in workflow


def _cron_hours(workflow: str) -> list[int]:
    lines = [ln.strip() for ln in workflow.splitlines() if ln.strip().startswith("- cron:")]
    return [int(ln.split('"')[1].split()[1]) for ln in lines]


def test_a_run_lands_after_both_closes(workflow: str) -> None:
    """22:00 UTC: New York has closed (20:00-21:00) and so has Taipei (05:30)."""
    assert 22 in _cron_hours(workflow)


def test_a_catch_up_run_lands_after_the_us_bar_is_published(workflow: str) -> None:
    """Without it MU records nothing whenever the main run lands early.

    Yahoo leaves a US session's prices blank until roughly 01:30 UTC, so a run
    before that sees MU one session behind and the journal withholds it. The
    catch-up hour must sit after that and before the 05:30 UTC Taipei close,
    with room for the scheduler's drift.
    """
    assert any(2 <= hour <= 3 for hour in _cron_hours(workflow))
