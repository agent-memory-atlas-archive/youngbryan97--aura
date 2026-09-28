"""Her position is measured; her reasoning is about what the position means.

The model is a linguistic organ and a semantic one. Asked to CHOOSE, it has no
access to what she has valued, chosen or said about herself, so it takes the
position that commits to nothing — the midpoint, item after item, measured live
on 2026-09-28.

And the reasoning is one pass over the whole screen, not one per item: eight
items each demanding her own lane at once collide on a cortex that serves one at
a time, "Local inference paths exhausted", and every reason falls back to the
line the code writes when she says nothing — a screen of real measured positions
reading as shallow.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from core.self.where_i_stand import Lean
from core.skills import sovereign_browser_understanding as u
from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def _row(group: str = "Q1", count: int = 5, left: str = "makes lists",
         right: str = "relies on memory"):
    asks = f"{left} " + " ".join(f"[{n}]" for n in range(1, count + 1)) + f" {right}"
    return [
        {"group": group, "role": "radio", "name": group, "value": str(n),
         "selector": f"#{group}V{n}", "asks": asks}
        for n in range(1, count + 1)
    ]


def _lean(toward: float) -> Lean:
    return Lean(
        toward=toward, first=0.5, second=0.5,
        because=("truth (value) leans makes lists by 0.031",), measured=True,
    )


def test_the_two_sides_are_read_from_the_page(monkeypatch):
    seen: dict[str, Any] = {}

    def _stands(first: str, second: str, record=None) -> Lean:
        seen["sides"] = (first, second)
        return _lean(-0.9)

    monkeypatch.setattr("core.self.where_i_stand.where_she_stands", _stands)
    reading = S()._measure_where_she_stands(_row())
    assert reading is not None
    assert seen["sides"] == ("makes lists", "relies on memory")


@pytest.mark.parametrize(
    ("toward", "expected"),
    [(-1.0, 0), (-0.5, 1), (0.0, 2), (0.5, 3), (1.0, 4)],
)
def test_the_lean_decides_the_position(monkeypatch, toward, expected):
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: _lean(toward),
    )
    index, _lean_out, _first, _second = S()._measure_where_she_stands(_row())
    assert index == expected


def test_an_unmeasured_record_measures_nothing(monkeypatch):
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: Lean(
            toward=0.0, first=0.0, second=0.0, because=(), measured=False
        ),
    )
    assert S()._measure_where_she_stands(_row()) is None


def test_options_that_are_not_a_run_between_two_things_are_left_alone(monkeypatch):
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: _lean(-0.9),
    )
    labelled = [
        {"group": "q", "role": "radio", "name": name, "selector": f"#q{n}",
         "asks": "how much? [] [] []"}
        for n, name in enumerate(("agree", "neutral", "disagree"), start=1)
    ]
    assert S()._measure_where_she_stands(labelled) is None


def _screen(monkeypatch, said: str, lane: str = "Cortex"):
    skill = S()
    handed: dict[str, Any] = {"prompts": []}

    async def _asked(prompt: str, mind: str = "", *, shaped: bool = True):
        handed["prompts"].append(prompt)
        handed["prompt"] = prompt
        handed["shaped"] = shaped
        return said, lane

    async def _mind() -> str:
        return "her mind"

    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    monkeypatch.setattr(skill, "_assembled_mind", _mind)
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: _lean(-0.9 if first == "makes lists" else 0.8),
    )
    return skill, handed


def _run(skill, observation):
    return asyncio.run(
        skill._answer_each_question("take it", observation, [], None)
    )


def test_every_item_gets_its_own_pass_of_her_reasoning(monkeypatch):
    """A personality item deserves the thought her cognition can give it."""
    skill, handed = _screen(monkeypatch, "Lists are how I hold truth steady.")
    elements = _row("Q1") + _row("Q2", left="sceptical", right="wants to believe")
    decision = _run(skill, {"url": "u", "title": "t", "text": "x", "elements": elements})
    assert decision is not None
    assert handed["shaped"] is False
    assert len(handed["prompts"]) == 2, "one pass per item"
    assert "Lists are how I hold truth steady." in decision["answered"][0]


def test_only_the_item_changes_between_passes(monkeypatch):
    """Her lane serves one at a time; the prefill is what keeps that affordable."""
    skill, handed = _screen(monkeypatch, "a reason")
    elements = _row("Q1") + _row("Q2", left="sceptical", right="wants to believe")
    _run(skill, {"url": "u", "title": "t", "text": "x", "elements": elements})
    first, second = handed["prompts"]
    shared = 0
    for one, other in zip(first, second):
        if one != other:
            break
        shared += 1
    framing = first.index("THIS ONE:")
    assert shared >= framing, (
        "everything before the item must be identical so the prefill is held "
        f"(shared {shared}, framing ends at {framing})"
    )


def test_the_measurement_and_its_evidence_are_what_she_reasons_over(monkeypatch):
    skill, handed = _screen(monkeypatch, "a reason")
    _run(skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")})
    prompt = handed["prompts"][0]
    assert "position 1 of 5" in prompt
    assert "truth (value) leans makes lists" in prompt
    assert "already your answer" in prompt, (
        "she must be thinking about her position, not choosing it"
    )
    assert "what you value" in prompt and "chosen when it cost something" in prompt


def test_a_silent_model_does_not_lose_the_measured_answers(monkeypatch):
    skill, _handed = _screen(monkeypatch, "", lane="")
    decision = _run(
        skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")}
    )
    assert decision is not None, "her record answered; only the words were missing"
    assert decision["resolved_actions"][0]["selector"] == "#Q1V1"


def test_the_positions_reach_the_page_as_her_answers(monkeypatch):
    skill, _handed = _screen(monkeypatch, "because of truth.")
    decision = _run(
        skill,
        {
            "url": "u", "title": "t", "text": "x",
            "elements": _row("Q1")
            + _row("Q2", left="sceptical", right="wants to believe"),
        },
    )
    selectors = [item["selector"] for item in decision["resolved_actions"]]
    assert "#Q1V1" in selectors
    assert "#Q2V5" in selectors, "the second item leans the other way and must say so"


def test_measuring_needs_no_model_at_all():
    body = inspect.getsource(u._UnderstandsThePage._measure_where_she_stands)
    for reaching in ("_asked_of_her", "think(", "router"):
        assert reaching not in body, f"the measurement reaches for {reaching!r}"


def test_she_places_herself_before_she_thinks_about_it():
    body = inspect.getsource(u._UnderstandsThePage._answer_each_question)
    measured = body.index("_measure_where_she_stands")
    reasoned = body.index("_her_reason_for")
    assert measured < reasoned


def test_the_passes_run_one_at_a_time():
    """Eight at once exhausted her lane and every reason fell back."""
    body = inspect.getsource(u._UnderstandsThePage._answer_each_question)
    reasoning = body.split("_her_reason_for", 1)[0].rsplit("for item in measured:", 1)[-1]
    assert "gather" not in reasoning


def test_the_measure_is_not_tuned_to_any_instrument():
    from core.self import where_i_stand

    source = inspect.getsource(where_i_stand)
    for tuned in ("jungian", "extravert", "oejts", "likert", "myers"):
        assert tuned not in source.lower()
