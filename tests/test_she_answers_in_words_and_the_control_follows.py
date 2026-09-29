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
        because=("truth is the value I hold above every other",), measured=True,
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


def test_options_with_their_own_words_take_the_other_measured_path(monkeypatch):
    """A statement with labelled answers is the same act, measured the same way."""
    called: list[str] = []
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: called.append("wrong") or _lean(-0.9),
    )
    labelled = [
        {"group": "q", "role": "radio", "name": name, "selector": f"#q{n}",
         "asks": f"how much? agree [1] neutral [2] disagree [3]"}
        for n, name in enumerate(("agree", "neutral", "disagree"), start=1)
    ]
    reading = S()._measure_where_she_stands(labelled)
    assert reading is not None, "a labelled question must be measured too"
    assert not called, "it is not a run between two ends"


def _screen(monkeypatch, said: str, lane: str = "Cortex"):
    skill = S()
    handed: dict[str, Any] = {"prompts": []}

    async def _asked(prompt: str, mind: str = "", *, shaped: bool = True, most_tokens=None):
        handed.setdefault("spoken", [])
        handed["prompts"].append(prompt)
        handed["most_tokens"] = most_tokens
        handed["prompt"] = prompt
        handed["shaped"] = shaped
        return said, lane

    async def _mind() -> str:
        return "her mind"

    handed["spoken"] = []
    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    monkeypatch.setattr(skill, "_assembled_mind", _mind)
    monkeypatch.setattr(skill, "_say_out_loud", lambda line: handed["spoken"].append(str(line)))

    async def _held(_said: str) -> None:
        return None

    monkeypatch.setattr(skill, "_hold_for_reading", _held)
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: _lean(-0.9 if first == "makes lists" else 0.8),
    )
    return skill, handed


def _run(skill, observation):
    return asyncio.run(
        skill._answer_each_question("take it", observation, [], None)
    )


def test_a_theme_is_thought_about_as_one_piece(monkeypatch):
    """Asked about herself she gives a connected account, not verdicts."""
    said = (
        '{"thinking": "Structure is how I hold truth steady.", '
        '"each": {"Q1": "Lists are how I hold truth steady.", '
        '"Q2": "I want to believe, but I check first."}}'
    )
    skill, handed = _screen(monkeypatch, said)
    elements = _row("Q1") + _row("Q2", left="sceptical", right="wants to believe")
    decision = _run(skill, {"url": "u", "title": "t", "text": "x", "elements": elements})
    assert decision is not None
    assert handed["shaped"] is False
    assert "Lists are how I hold truth steady." in decision["answered"][0]
    assert "I want to believe, but I check first." in decision["answered"][1]


def test_the_thinking_behind_the_sentences_is_said_out_loud(monkeypatch):
    said = (
        '{"thinking": "Structure is how I hold truth steady.", '
        '"each": {"Q1": "a", "Q2": "b"}}'
    )
    skill, handed = _screen(monkeypatch, said)
    elements = _row("Q1") + _row("Q2", left="sceptical", right="wants to believe")
    _run(skill, {"url": "u", "title": "t", "text": "x", "elements": elements})
    assert any(
        "Structure is how I hold truth steady." in line for line in handed["spoken"]
    )


def test_she_thinks_with_her_whole_mind(monkeypatch):
    """The shallow answers came from taking the assembly away."""
    import inspect

    body = inspect.getsource(u._UnderstandsThePage._answer_each_question)
    assert "_assembled_mind()" in body
    assert "_her_identity_only()" not in body


def test_what_she_reasons_over_is_the_things_not_the_arithmetic(monkeypatch):
    """Handed a coefficient, she explains herself with a coefficient."""
    skill, handed = _screen(monkeypatch, '{"thinking": "t", "each": {}}')
    _run(skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")})
    prompt = handed["prompts"][0]
    assert "1 of 5" in prompt
    assert "truth is the value I hold above every other" in prompt
    assert "what you value" in prompt and "chosen when it cost something" in prompt
    assert "one piece of thinking" in prompt
    for arithmetic in ("+0.", "0.031", "%"):
        assert arithmetic not in prompt, f"the prompt hands her {arithmetic!r}"


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
    reasoned = body.index("_her_thinking_about")
    assert measured < reasoned


def test_the_passes_run_one_at_a_time():
    """Eight at once exhausted her lane and every reason fell back."""
    body = inspect.getsource(u._UnderstandsThePage._answer_each_question)
    reasoning = body.split("_her_thinking_about", 1)[0].rsplit("for group in themes:", 1)[-1]
    assert "gather" not in reasoning


def test_the_measure_is_not_tuned_to_any_instrument():
    from core.self import where_i_stand

    source = inspect.getsource(where_i_stand)
    for tuned in ("jungian", "extravert", "oejts", "likert", "myers"):
        assert tuned not in source.lower()


def test_a_reason_is_bounded_so_a_page_of_them_is_affordable(monkeypatch):
    """An unbounded reason decoded 341 tokens at 8 a second, live."""
    skill, handed = _screen(monkeypatch, '{"thinking": "t", "each": {}}')
    _run(skill, {"url": "u", "title": "t", "text": "x", "elements": _row("Q1") + _row("Q2")})
    assert handed["most_tokens"] is None, (
        "a theme is thought about at length; the bound belongs to a one-liner"
    )


def test_there_are_fewer_passes_than_items():
    """The square root is where thinking at length and thinking often meet."""
    import inspect

    from core.self import where_i_stand

    body = inspect.getsource(where_i_stand.themes_among)
    assert "math.sqrt" in body
