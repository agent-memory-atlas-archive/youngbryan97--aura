"""A commentary a person asked for is for them to read.

The pursuit narrated a decision and acted on it in the same breath, so the next
line replaced the last before it could be read and a watcher got a blur they
could reconstruct only from the log afterwards — which is the thing narration
exists to avoid. Asked for directly on 2026-09-28: "narrate each choice and
pause to give time to read, ~5 seconds".
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from core.agency.reading_pace import AT_LEAST_S, AT_MOST_S, time_to_read
from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


def test_nothing_to_read_is_no_wait():
    assert time_to_read("") == 0.0
    assert time_to_read("   ") == 0.0


def test_a_short_line_still_gets_time_to_be_seen():
    assert time_to_read("Clicking Next.") == AT_LEAST_S


def test_a_longer_line_gets_the_time_it_takes_to_read():
    hundred = " ".join(["word"] * 100)
    assert time_to_read(hundred) > AT_LEAST_S


def test_one_enormous_line_cannot_stall_the_run():
    assert time_to_read(" ".join(["word"] * 5000)) == AT_MOST_S


def test_the_pace_can_be_turned_off_for_a_run_nobody_watches(monkeypatch):
    monkeypatch.setenv("AURA_NARRATION_PACE", "0")
    assert time_to_read("Clicking Next.") == 0.0


def test_a_broken_setting_is_ignored_rather_than_obeyed(monkeypatch):
    monkeypatch.setenv("AURA_NARRATION_PACE", "soon")
    assert time_to_read("Clicking Next.") == AT_LEAST_S


def test_the_pursuit_waits_after_it_narrates(monkeypatch):
    """The hold is inside the loop, between the saying and the acting."""
    waited: list[str] = []

    async def _held(said: str) -> None:
        waited.append(said)

    class _Page:
        async def observe(self, **_kw: Any) -> dict[str, Any]:
            return {
                "url": "https://example.test/survey",
                "title": "A survey",
                "text": "Question one.",
                "elements": [{"role": "button", "name": "Next", "selector": "#next"}],
            }

        async def close(self) -> None:
            return None

    skill = SovereignBrowserSkill()

    async def _understood(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"here": "a survey", "to_progress": "answer", "done_when": "submitted"}

    async def _decided(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"error": "empty_decision", "raw": ""}

    monkeypatch.setattr(skill, "_understand_page", _understood)
    monkeypatch.setattr(skill, "_decide_next_actions", _decided)
    monkeypatch.setattr(skill, "_asks_about_the_one_answering", lambda *_a: False)
    monkeypatch.setattr(skill, "_narrate", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_say_out_loud", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_remember_the_place", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_recall_about", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_retain_stated_positions", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_hold_for_reading", _held)

    asyncio.run(skill._handle_pursue(_Page(), None, "take the survey", 4))
    assert waited, "the loop narrated and acted without leaving the line up"
    assert any("empty_decision" in said for said in waited)


def test_the_wait_is_real_time_and_not_a_promise(monkeypatch):
    monkeypatch.setenv("AURA_NARRATION_PACE", "0.02")
    started = time.monotonic()
    asyncio.run(SovereignBrowserSkill._hold_for_reading("Clicking Next."))
    assert time.monotonic() - started >= AT_LEAST_S * 0.02 * 0.5


def test_a_game_keeps_its_own_tempo():
    """A loop with a clock of its own is not paced by a reader.

    Holding a 2048 move for five seconds would be playing the narration rather
    than the game; the page waits for her click and can afford it.
    """
    import inspect

    from core.skills import screen_pursuit_looking

    assert "reading_pace" not in inspect.getsource(screen_pursuit_looking), (
        "a live game's loop must not be paced by a reader"
    )
