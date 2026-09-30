"""The goal lock is held for the write, not for preparing what to write.

Every one of the eight stalls in the 36 hours to 29 September 04:33 was the event
loop inside `GoalEngine._fetch_records`, between 5.4 and 8.1 seconds, and that
read waits on `self._lock`. A 12 ms query cannot spend eight seconds; what it can
do is wait behind a writer that holds the lock across its own work.

Two things were in there that need no lock. `_write_record` serialised her
evidence and metadata with four `json.dumps` calls, on objects that belong to the
caller. `quarantine_transient_foreground_goals` found, read and compiled three
modules — a disk read inside a critical section every reader waits on.

This holds the structure rather than a duration: a timing threshold on a loaded
host is what produced these stall reports in the first place.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from core.goals import goal_engine

pytestmark = pytest.mark.unit


def _code_of(method) -> str:
    """A method's source with its comments removed.

    A comment that quotes `with self._lock:` — and the ones explaining these
    fixes do — would otherwise split a partition in the wrong place.
    """
    return "\n".join(
        line for line in inspect.getsource(method).splitlines()
        if not line.lstrip().startswith("#")
    )


def _locked_sections():
    tree = ast.parse(Path(inspect.getsourcefile(goal_engine)).read_text())
    functions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node in ast.walk(tree):
        if not isinstance(node, ast.With):
            continue
        if not any("self._lock" in ast.unparse(item.context_expr) for item in node.items):
            continue
        owner = max(
            (f for f in functions if f.lineno < node.lineno <= (f.end_lineno or 0)),
            key=lambda f: f.lineno,
            default=None,
        )
        yield (owner.name if owner else "?"), node


def test_no_section_holding_the_goal_lock_imports_a_module():
    offenders = {
        name for name, node in _locked_sections()
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node))
    }
    assert not offenders, offenders


def test_the_write_serialises_before_it_takes_the_lock():
    source = _code_of(goal_engine.GoalEngine._write_record)
    before, _, after = source.partition("with self._lock:")
    assert before.count("json.dumps") == 4, "her evidence and metadata are prepared first"
    assert "json.dumps" not in after
    assert "written = {" in before


def test_the_lock_still_covers_the_write_and_its_commit():
    source = _code_of(goal_engine.GoalEngine._write_record)
    _, _, after = source.partition("with self._lock:")
    assert "self._conn.execute(" in after
    assert "self._conn.commit()" in after


def test_the_quarantine_reads_its_modules_before_the_lock():
    source = _code_of(goal_engine.GoalEngine.quarantine_transient_foreground_goals)
    before, _, after = source.partition("with self._lock:")
    assert "has_explicit_durable_binding" in before
    assert "standing_objective_rejection_reason" in before
    assert "import" not in after


def test_the_read_holds_the_lock_for_the_read_alone():
    source = _code_of(goal_engine.GoalEngine._fetch_records)
    _, _, after = source.partition("with self._lock:")
    # The query and the rows it returns, and nothing else.
    assert "self._conn.execute(" in after
    assert "json.dumps" not in after


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])


def test_the_quarantine_diagnoses_outside_the_lock():
    """It read every active goal, then ran `_row_to_record`, `to_dict` and two
    standing-objective diagnoses for each one, then an UPDATE for each it
    condemned — all in one critical section that every reader waits on."""
    source = _code_of(goal_engine.GoalEngine.quarantine_transient_foreground_goals)
    read, _, rest = source.partition("with self._lock:")
    diagnose, _, write = rest.partition("with self._lock:")
    assert "fetchall()" in diagnose.split("condemned")[0]
    # The per-row work is between the two locks.
    for outside in ("_row_to_record", "to_dict()", "is_transient_foreground_projection"):
        assert outside in diagnose, outside
        assert outside not in write, outside
    # And the writes are under the second one.
    assert "UPDATE goals" in write


def test_a_row_whose_status_moved_is_left_alone():
    """The guard the single critical section did not need: a row diagnosed from a
    status it is no longer in would be abandoned on stale evidence."""
    source = _code_of(goal_engine.GoalEngine.quarantine_transient_foreground_goals)
    _, _, write = source.partition("condemned.append")
    assert "WHERE id = ? AND status = ?" in write
    assert ".rowcount" in write
    assert "if changed:" in write
