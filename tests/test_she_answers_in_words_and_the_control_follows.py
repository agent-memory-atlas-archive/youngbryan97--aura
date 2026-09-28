"""Her cognition produces speech, so she is asked for her placement in speech.

LIVE 2026-09-28 18:06: the decision went to her own lane, twice, and came back
"The user wants me to take the Open Extended Jungian Type Scales test on
openpsychometrics.org. Before starting, I need..." — reported as
`unparsable_decision`, one round, no progress. A cognitive cycle is built to
produce what she would say; asking it for a structured object gets a sentence
about the task.

So she is asked for the thing she is actually being asked for — where she puts
herself — and the control follows from the position she named. Reading a number
she wrote is not interpreting her answer.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def _row(count: int = 5):
    asks = "makes lists " + " ".join(f"[{n}]" for n in range(1, count + 1)) + " relies on memory"
    return [
        {"group": "Q1", "role": "radio", "name": "Q1", "value": str(n),
         "selector": f"#Q1V{n}", "asks": asks}
        for n in range(1, count + 1)
    ]


def test_a_position_she_named_is_found():
    assert S._the_position_she_named("I put myself at position 2 of 5.", 5) == 1
    assert S._the_position_she_named("about a 4 out of 5", 5) == 3
    assert S._the_position_she_named("the midpoint, 3 of 5", 5) == 2


def test_no_position_named_is_not_guessed_at():
    assert S._the_position_she_named("I lean strongly toward lists", 5) is None


def test_a_position_the_run_does_not_have_is_refused():
    assert S._the_position_she_named("position 9 of 5", 5) is None
    assert S._the_position_she_named("position 0", 5) is None


def _skill(monkeypatch, said: str, lane: str = "Cortex"):
    skill = S()

    async def _asked(prompt: str, mind: str = "", *, shaped: bool = True):
        assert shaped is False, "a placement is said, not composed"
        return said, lane

    async def _mind() -> str:
        return "her mind"

    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    monkeypatch.setattr(skill, "_assembled_mind", _mind)
    return skill


def test_the_control_follows_from_what_she_said(monkeypatch):
    skill = _skill(monkeypatch, "I make lists for everything, so position 1 of 5.")
    placed = asyncio.run(
        skill._where_she_puts_herself(
            "take it", {"url": "u", "title": "t", "text": "x", "elements": _row()},
            _row(), None,
        )
    )
    assert placed is not None
    assert placed["selector"] == "#Q1V1"
    assert "makes lists" in placed["said"]


def test_the_far_end_is_reachable(monkeypatch):
    skill = _skill(monkeypatch, "Memory, entirely. Position 5 of 5.")
    placed = asyncio.run(
        skill._where_she_puts_herself(
            "take it", {"url": "u", "title": "t", "text": "x", "elements": _row()},
            _row(), None,
        )
    )
    assert placed["selector"] == "#Q1V5"


def test_a_placement_from_another_lane_is_not_taken(monkeypatch):
    skill = _skill(monkeypatch, "position 2 of 5", lane="Brainstem")
    placed = asyncio.run(
        skill._where_she_puts_herself(
            "take it", {"url": "u", "title": "t", "text": "x", "elements": _row()},
            _row(), None,
        )
    )
    assert placed is None


def test_a_placement_that_names_nothing_falls_back(monkeypatch):
    skill = _skill(monkeypatch, "I feel quite strongly about this.")
    placed = asyncio.run(
        skill._where_she_puts_herself(
            "take it", {"url": "u", "title": "t", "text": "x", "elements": _row()},
            _row(), None,
        )
    )
    assert placed is None, "the shaped path must get its turn"


def test_asking_in_words_comes_before_asking_in_a_shape():
    import inspect

    body = inspect.getsource(S._answer_each_question)
    words = body.index("_where_she_puts_herself")
    shape = body.index("_decide_next_actions")
    assert words < shape
