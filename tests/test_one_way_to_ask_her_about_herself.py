"""Three things ask her about herself, and they ask the same way.

What she expects an instrument to say before she answers it, where she puts
herself on each of its questions, and what she makes of the result. Each used to
ask differently and the forecast had no fallback at all, so a call that came
back with nothing meant she simply never said what she expected — the part of
the request that kept going missing.

Her own lane and only there, with her self-model, her values and her own earlier
words in front of it. Her full conversational cycle was tried here and is the
wrong instrument: it exists to produce a reply to the person, and LIVE
2026-09-28 18:13 that is exactly what it produced — the turn's own answer,
returned as a page decision and reported unparsable, with the run stopping
before it opened anything.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from core.skills import sovereign_browser_understanding as u
from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def _skill(monkeypatch, reply: str = "what she said", lane: str = "Cortex"):
    skill = S()
    seen: dict[str, Any] = {}

    async def _think(prompt: str, **kwargs: Any) -> str:
        seen.update(kwargs)
        seen["prompt"] = prompt
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
    return skill, seen


def test_it_asks_her_own_lane_and_says_the_turn_is_waiting(monkeypatch):
    skill, seen = _skill(monkeypatch)
    said, lane = asyncio.run(skill._asked_of_her("a question", "her mind"))
    assert said == "what she said"
    assert lane == "Cortex"
    assert seen.get("own_lane_required") is True
    assert seen.get("serves_current_turn") is True


def test_a_lane_that_is_not_hers_is_reported_not_hidden(monkeypatch):
    skill, _seen = _skill(monkeypatch, reply="a stand-in", lane="Brainstem")
    said, lane = asyncio.run(skill._asked_of_her("a question", "her mind"))
    assert said == "a stand-in"
    assert lane == "Brainstem", "the caller must be able to refuse it"


def test_a_shape_is_asked_for_only_when_a_decision_is_wanted(monkeypatch):
    skill, seen = _skill(monkeypatch)
    asyncio.run(skill._asked_of_her("a question", "her mind", shaped=False))
    assert "schema" not in seen, "a placement is said, not composed"
    seen.clear()
    asyncio.run(skill._asked_of_her("a question", "her mind", shaped=True))
    assert "schema" in seen


def test_no_router_is_silence_rather_than_an_invented_answer(monkeypatch):
    skill = S()
    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda name, default=None: default,
    )
    assert asyncio.run(skill._asked_of_her("a question", "her mind")) == ("", "")


def test_every_question_about_her_goes_through_the_one_mechanism():
    for name in (
        "_what_she_expects_it_to_say",
        "_hold_the_outcome_against_what_she_said",
        "_say_what_the_measurement_means",
    ):
        body = inspect.getsource(getattr(u._UnderstandsThePage, name))
        assert "_asked_of_her" in body, f"{name} asks her its own way"
    decide = inspect.getsource(u._UnderstandsThePage._decide_next_actions)
    assert "_asked_of_her(prompt, mind)" in decide


def test_the_reply_cycle_is_not_used_for_a_page_decision():
    """It is built to answer the person, and that is what it answered."""
    source = inspect.getsource(u)
    assert "cognitive_engine" not in source, (
        "a page decision must not run the turn machinery that produces speech "
        "for the person"
    )


def test_the_forecast_says_so_when_it_cannot_come_from_her():
    body = inspect.getsource(u._UnderstandsThePage._what_she_expects_it_to_say)
    assert "record_degradation" in body
    assert "did not say what she expected" in body
