"""A task that begins with a forecast does not end when the last control is pressed.

The person asked for the forecast so it could be held against the outcome; a
forecast never checked was decoration. `result["concluded"]` was read where the
pursuit's account is written and written by nothing, so a run that ended on a
result page said nothing about the result — the account finished with the raw
tail of the page.

General, not specific to any site or task: it runs wherever she said something
before she began, and asks only about her own earlier claim and what is in front
of her now.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


class _Page:
    async def observe(self, **_kw: Any) -> dict[str, Any]:
        return {
            "url": "https://example.test/results",
            "title": "Your result",
            "text": "Your result is ENTP.",
            "elements": [{"role": "button", "name": "Next", "selector": "#next"}],
        }

    async def close(self) -> None:
        return None


def _skill(monkeypatch, asked: list[str], landed: bool = True):
    skill = SovereignBrowserSkill()
    rounds = {"n": 0}

    async def _understood(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"here": "a result", "to_progress": "read it", "done_when": "read"}

    async def _decided(*_a: Any, **_k: Any) -> dict[str, Any]:
        rounds["n"] += 1
        if rounds["n"] == 1 and landed:
            return {"actions": [{"index": 0, "type": "click"}], "why": "press it", "done": False}
        return {"done": True, "actions": [], "why": "finished"}

    async def _interacted(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"ok": True, "results": [{"ok": True}]}

    async def _concluded(goal: str, said_before: str, observation: Any, mind: str) -> str:
        asked.append(f"{said_before}|{observation.get('text')}")
        return "I said ENTP and the page says ENTP, which reads true to me."

    monkeypatch.setattr(skill, "_understand_page", _understood)
    monkeypatch.setattr(skill, "_decide_next_actions", _decided)
    monkeypatch.setattr(skill, "_handle_interact", _interacted)
    monkeypatch.setattr(skill, "_asks_about_the_one_answering", lambda *_a: False)
    monkeypatch.setattr(skill, "_narrate", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_narrate_decision", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_say_out_loud", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_remember_the_place", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_recall_about", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_retain_stated_positions", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_hold_the_outcome_against_what_she_said", _concluded)

    async def _held(_said: str) -> None:
        return None

    monkeypatch.setattr(skill, "_hold_for_reading", _held)
    return skill


def test_the_outcome_is_held_against_what_she_said(monkeypatch):
    asked: list[str] = []
    skill = _skill(monkeypatch, asked)
    result = asyncio.run(
        skill._handle_pursue(
            _Page(), None, "take the test", 6, said_before="I expect ENTP."
        )
    )
    assert asked, "the run finished without checking its own forecast"
    assert "I expect ENTP." in asked[0]
    assert "Your result is ENTP." in asked[0], (
        "she must be shown the finished page, not the one she started on"
    )
    assert "reads true to me" in result["concluded"]


def test_a_run_with_no_forecast_has_nothing_to_hold(monkeypatch):
    asked: list[str] = []
    skill = _skill(monkeypatch, asked)
    result = asyncio.run(skill._handle_pursue(_Page(), None, "take the test", 6))
    assert result["concluded"] == "" or not asked or asked[0].startswith("|")


def test_a_run_that_did_nothing_concludes_nothing(monkeypatch):
    """There is no outcome to hold a forecast against when nothing happened."""
    asked: list[str] = []
    skill = _skill(monkeypatch, asked, landed=False)
    result = asyncio.run(
        skill._handle_pursue(
            _Page(), None, "take the test", 6, said_before="I expect ENTP."
        )
    )
    assert not asked
    assert result["concluded"] == ""


def test_the_conclusion_reaches_the_account_the_person_reads():
    """`_pursuit_account` reads `concluded`; the delegation must carry it."""
    import inspect

    from core.skills import desktop_task

    source = inspect.getsource(desktop_task._DelegatesAPageObjective) if hasattr(
        desktop_task, "_DelegatesAPageObjective"
    ) else inspect.getsource(desktop_task)
    assert 'report.get("concluded")' in source, (
        "the pursuit's conclusion must reach the desktop task's result, or the "
        "reply ends on the raw tail of the page again"
    )
