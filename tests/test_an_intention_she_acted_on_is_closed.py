"""An intention she acted on is closed with what came of it.

Intend, act, observe and revise is the loop (core/agency/intention_loop.py),
and the subject driver stopped after observing. On the seed-7 run at 27dc1dda9
all 3,608 intentions it declared were still in progress at the end, so her open
intentions were everything she had ever meant to do, none of it finished.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.agency.intention_loop import IntentionLoop, IntentionStatus
from core.subject.driver import SubjectRuntime


def _runtime(tmp_path) -> SimpleNamespace:
    return SimpleNamespace(
        _intentions=IntentionLoop(db_path=str(tmp_path / "intentions.db")),
        ACTION_EXPECTATIONS=SubjectRuntime.ACTION_EXPECTATIONS,
    )


def test_an_act_that_worked_closes_its_intention_as_completed(tmp_path):
    runtime = _runtime(tmp_path)
    SubjectRuntime._through_the_intention_loop(runtime, "write the plan down", True, "self", "append_log")
    assert runtime._intentions.get_open_intentions() == []
    closed = runtime._intentions.get_revision_history()
    assert [rec.status for rec in closed] == [IntentionStatus.COMPLETED]


def test_an_act_that_failed_closes_its_intention_as_failed(tmp_path):
    runtime = _runtime(tmp_path)
    SubjectRuntime._through_the_intention_loop(runtime, "paint the panel", False, "self", "paint_panel")
    assert runtime._intentions.get_open_intentions() == []
    assert [rec.status for rec in runtime._intentions.get_revision_history()] == [IntentionStatus.FAILED]


def test_many_acts_leave_nothing_open(tmp_path):
    runtime = _runtime(tmp_path)
    for turn in range(20):
        SubjectRuntime._through_the_intention_loop(runtime, f"act {turn}", turn % 3 != 0, "self", "append_log")
    assert runtime._intentions.get_open_intentions() == []
