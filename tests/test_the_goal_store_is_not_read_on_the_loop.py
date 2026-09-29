"""A goal written from a coroutine is projected from a thread, and reading goals never builds on the loop.

LIVE, 29 September 2026: eight event-loop stalls in one night, 5.4 to 8.1
seconds each, every one inside `GoalEngine._fetch_records`. The paths were
`track_dispatch` and `update_task_lifecycle`, which re-projected the goals onto
her state inline after each write, and initiative synthesis, the mind tick and
the task engine, which asked for her own goals without the external ones. That
request skipped the stale-while-revalidate snapshot the hot path was given in
July, so each one built a fresh snapshot, reconciliation writes included, on
the loop.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.goals.goal_engine import GoalEngine


def _engine(tmp_path: Path) -> tuple[GoalEngine, list[int]]:
    engine = GoalEngine(db_path=str(tmp_path / "goals.db"))
    built_on: list[int] = []
    original = engine.build_snapshot

    def recording(*args, **kwargs):
        built_on.append(threading.get_ident())
        return original(*args, **kwargs)

    engine.build_snapshot = recording  # type: ignore[method-assign]
    engine._state_repo = SimpleNamespace(_current=SimpleNamespace(cognition=SimpleNamespace(
        active_goals=[], current_objective=None, current_origin="",
    )))
    return engine, built_on


def test_a_goal_written_from_a_coroutine_is_projected_off_the_loop(tmp_path: Path) -> None:
    engine, built_on = _engine(tmp_path)

    async def write() -> int:
        await engine.add_goal("tidy the notes", priority=0.7)
        return threading.get_ident()

    loop_thread = asyncio.run(write())
    assert built_on, "the projection never read the store"
    assert loop_thread not in built_on, "the snapshot was built on the event loop"
    cognition = engine.state_repo._current.cognition
    assert [goal["goal"] for goal in cognition.active_goals] == ["tidy the notes"]


def test_her_own_goals_are_read_from_the_snapshot_once_it_exists(tmp_path: Path) -> None:
    engine, built_on = _engine(tmp_path)
    asyncio.run(engine.add_goal("tidy the notes", priority=0.7))
    engine.get_active_goals(limit=5)  # warms the snapshot
    before = len(built_on)

    async def read() -> int:
        engine.get_active_goals(limit=5, include_external=False, actionable_only=True)
        return threading.get_ident()

    loop_thread = asyncio.run(read())
    assert loop_thread not in built_on[before:], "asking for her own goals built a snapshot on the loop"


def test_a_caller_that_must_see_its_write_can_ask_for_it_fresh(tmp_path: Path) -> None:
    engine, built_on = _engine(tmp_path)
    engine.get_active_goals(limit=5)
    engine.sync_task_plan(SimpleNamespace(goal="file the receipts", status="queued", steps=[], plan_id="p1"))
    before = len(built_on)
    fresh = engine.get_active_goals(limit=5, include_external=False, fresh=True)
    assert len(built_on) == before + 1
    assert "file the receipts" in [goal.get("objective") for goal in fresh]


@pytest.mark.parametrize("limit", [1, 6])
def test_the_projection_carries_what_leaving_each_goal_costs(tmp_path: Path, limit: int) -> None:
    """The one owner of `cognition.active_goals`. The task engine wrote a raw list over it."""
    engine, _ = _engine(tmp_path)
    asyncio.run(engine.add_goal("tidy the notes", priority=0.7))
    cognition = SimpleNamespace(active_goals=[], current_objective=None, current_origin="")
    engine.project_onto(cognition, limit=limit)
    assert cognition.active_goals and all("urgency" in goal for goal in cognition.active_goals)
    assert cognition.current_objective == "tidy the notes"
