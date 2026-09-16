"""The local daily loop, and the two things it must never get wrong.

`scripts/daily_update.py` commits and pushes to `main` unattended, from a
working tree rather than a clean runner checkout. Two failures there are not
recoverable by review, so they are pinned here:

1. A rewritten journal row. Every row was written before its outcome existed;
   that is the entire basis for trusting the number. A row that changed was
   recomputed with hindsight and is indistinguishable from an honest one
   afterwards, so the check has to fire before the commit.
2. Unrelated staged work swept into a data commit and pushed to `main`.
   `git commit` commits the index, not the paths you just added - the script
   relies on `--only` to scope it, and that reliance is tested against real
   git rather than assumed.

The download itself is not covered: it needs the network, which is the one
path CI could never test either.
"""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "daily_update.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("daily_update", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


daily_update = _load_script()


# --------------------------------------------------------------------------- #
# The journal may only ever grow
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("previous", "current", "expected"),
    [
        # An ordinary day: rows appended, nothing touched.
        ("h\na\nb\n", "h\na\nb\nc\n", True),
        # Nothing recorded (the feed is stale) - unchanged is still append-only.
        ("h\na\n", "h\na\n", True),
        # First ever run: nothing committed to compare against.
        ("", "h\na\n", True),
        # A row edited in place - the failure this exists to catch.
        ("h\na\nb\n", "h\na\nEDITED\n", False),
        # The header rewritten, e.g. a column renamed under the record.
        ("h\na\n", "h2\na\nb\n", False),
        # Rows dropped.
        ("h\na\nb\nc\n", "h\na\nb\n", False),
        # Truncated to nothing.
        ("h\na\n", "", False),
        # Re-ordered: same rows, different order, still a rewrite.
        ("h\na\nb\n", "h\nb\na\n", False),
    ],
)
def test_journal_is_append_only(previous: str, current: str, expected: bool) -> None:
    assert daily_update.journal_is_append_only(previous, current) is expected


def test_append_only_ignores_a_missing_trailing_newline() -> None:
    """A last line without \\n is the same row; it must not read as a rewrite."""
    assert daily_update.journal_is_append_only("h\na", "h\na\nb\n")


def test_rows_added_counts_only_the_new_lines() -> None:
    assert daily_update.rows_added("h\na\n", "h\na\nb\nc\n") == 2
    assert daily_update.rows_added("h\na\n", "h\na\n") == 0


# --------------------------------------------------------------------------- #
# git behaviour, against real git
# --------------------------------------------------------------------------- #
def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repo laid out like this one, on `main`."""
    _git("init", "-b", "main", cwd=tmp_path)
    _git("config", "user.email", "test@example.com", cwd=tmp_path)
    _git("config", "user.name", "Test", cwd=tmp_path)

    journal = tmp_path / "data" / "journal" / "forecasts.csv"
    journal.parent.mkdir(parents=True)
    journal.write_text("asof_date,symbol\n2026-09-11,MU\n", encoding="utf-8")
    prices = tmp_path / "data" / "prices" / "MU.csv"
    prices.parent.mkdir(parents=True)
    prices.write_text("date,close\n", encoding="utf-8")
    (tmp_path / "src.py").write_text("original\n", encoding="utf-8")

    _git("add", "-A", cwd=tmp_path)
    _git("commit", "-m", "initial", cwd=tmp_path)
    return tmp_path


def test_preflight_refuses_a_branch_it_must_not_commit_to(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A data commit on a feature branch is a day of forecasts quietly lost."""
    _git("checkout", "-b", "feature", cwd=repo)
    monkeypatch.chdir(repo)

    with pytest.raises(SystemExit) as excinfo:
        daily_update.preflight("main")

    assert "feature" in str(excinfo.value)


def test_preflight_accepts_the_expected_branch(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    daily_update.preflight("main")  # must not raise


def test_committed_journal_reads_the_version_in_head(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(repo)
    journal = repo / "data" / "journal" / "forecasts.csv"
    journal.write_text("asof_date,symbol\n2026-09-11,MU\n2026-09-16,MU\n", encoding="utf-8")

    committed = daily_update.committed_journal()

    # The working copy moved on; HEAD has not.
    assert committed == "asof_date,symbol\n2026-09-11,MU\n"
    assert daily_update.journal_is_append_only(committed, journal.read_text(encoding="utf-8"))


def test_committed_journal_is_empty_before_the_first_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _git("init", "-b", "main", cwd=tmp_path)
    monkeypatch.chdir(tmp_path)

    assert daily_update.committed_journal() == ""


def test_commit_only_leaves_unrelated_staged_work_out_of_the_commit(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The footgun: `git commit` commits the index, not the paths you added.

    Without `--only`, a developer's staged work in progress would ride along
    inside a "Update prices and forecast journal" commit and get pushed to
    `main`.
    """
    monkeypatch.chdir(repo)
    # Work in progress, staged and emphatically not meant for main.
    (repo / "src.py").write_text("half-finished refactor\n", encoding="utf-8")
    _git("add", "src.py", cwd=repo)
    # The day's data.
    (repo / "data" / "journal" / "forecasts.csv").write_text(
        "asof_date,symbol\n2026-09-11,MU\n2026-09-16,MU\n", encoding="utf-8"
    )

    result = daily_update.git(
        "commit", "--only", *daily_update.COMMITTED_PATHS, "-m", "data", check=False
    )
    assert result.returncode == 0, result.stderr

    committed = _git("show", "--name-only", "--format=", "HEAD", cwd=repo).stdout.split()
    assert "data/journal/forecasts.csv" in committed
    assert "src.py" not in committed, "staged work in progress was swept into the data commit"

    # And it is still staged, not silently dropped.
    staged = _git("diff", "--cached", "--name-only", cwd=repo).stdout.split()
    assert "src.py" in staged


def test_paths_are_dirty_sees_only_the_data_paths(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unchanged feed must report nothing to commit, however dirty the tree."""
    monkeypatch.chdir(repo)
    assert not daily_update.paths_are_dirty()

    (repo / "src.py").write_text("edited\n", encoding="utf-8")
    assert not daily_update.paths_are_dirty(), "a source edit is not a data change"

    (repo / "data" / "prices" / "MU.csv").write_text("date,close\n2026-09-16,1\n", encoding="utf-8")
    assert daily_update.paths_are_dirty()
