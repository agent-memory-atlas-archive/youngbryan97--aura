"""Three things ask her about herself, and they ask the same way.

What she expects an instrument to say before she answers it, how she answers
each of its questions, and what she makes of the result. Two of them had her
cognition with a direct call behind it, and the forecast had no fallback at all
— so a cycle that came back with nothing meant she simply never said what she
expected, which is the part of the request that kept going missing.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from core.skills import sovereign_browser_understanding as u
from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


class _Cycle:
    def __init__(self, content: str = "", lane: str = "Cortex") -> None:
        self.content = content
        self.lane = lane
        self.asked = 0

    async def __call__(self, prompt: str, *, shaped: bool = True):
        self.asked += 1
        return self.content, (self.lane if self.content else "")


def _skill(monkeypatch, cycle: _Cycle, reply: str = "", lane: str = "Cortex"):
    skill = S()
    seen: dict[str, Any] = {}

    async def _think(prompt: str, **kwargs: Any) -> str:
        seen.update(kwargs)
        sink = kwargs.get("_generation_metadata_sink")
        if isinstance(sink, dict):
            sink["endpoint"] = lane
        return reply

    class _Router:
        think = staticmethod(_think)

    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda name, default=None: _Router() if name == "llm_router" else default,
    )
    monkeypatch.setattr(skill, "_her_own_thinking_about_herself", cycle)
    return skill, seen


def test_her_cognition_is_asked_first(monkeypatch):
    cycle = _Cycle("what she thinks")
    skill, seen = _skill(monkeypatch, cycle)
    said, lane = asyncio.run(skill._asked_of_her("a question", "her mind"))
    assert cycle.asked == 1
    assert said == "what she thinks"
    assert lane == "Cortex"
    assert not seen, "the direct call ran even though her cognition answered"


def test_a_silent_cycle_falls_back_to_her_own_lane(monkeypatch):
    cycle = _Cycle("")
    skill, seen = _skill(monkeypatch, cycle, reply="from the lane")
    said, lane = asyncio.run(skill._asked_of_her("a question", "her mind"))
    assert said == "from the lane"
    assert lane == "Cortex"
    assert seen.get("own_lane_required") is True
    assert seen.get("serves_current_turn") is True


def test_the_fallback_is_shaped_only_when_a_decision_is_wanted(monkeypatch):
    cycle = _Cycle("")
    skill, seen = _skill(monkeypatch, cycle, reply="a sentence")
    asyncio.run(skill._asked_of_her("a question", "her mind", shaped=False))
    assert "schema" not in seen, "a forecast is a sentence, not an object"
    seen.clear()
    asyncio.run(skill._asked_of_her("a question", "her mind", shaped=True))
    assert "schema" in seen


def test_a_lane_that_is_not_hers_is_reported_not_hidden(monkeypatch):
    cycle = _Cycle("")
    skill, _seen = _skill(monkeypatch, cycle, reply="a stand-in", lane="Brainstem")
    said, lane = asyncio.run(skill._asked_of_her("a question", "her mind"))
    assert said == "a stand-in"
    assert lane == "Brainstem", "the caller must be able to refuse it"


def test_every_question_about_her_goes_through_the_one_mechanism():
    for name in (
        "_what_she_expects_it_to_say",
        "_hold_the_outcome_against_what_she_said",
    ):
        body = inspect.getsource(getattr(u._UnderstandsThePage, name))
        assert "_asked_of_her" in body, f"{name} asks her its own way"
    decide = inspect.getsource(u._UnderstandsThePage._decide_next_actions)
    assert "_asked_of_her(prompt, mind)" in decide


def test_the_forecast_says_so_when_it_cannot_come_from_her():
    """Silence was indistinguishable from having nothing to say."""
    body = inspect.getsource(u._UnderstandsThePage._what_she_expects_it_to_say)
    assert "record_degradation" in body
    assert "did not say what she expected" in body
