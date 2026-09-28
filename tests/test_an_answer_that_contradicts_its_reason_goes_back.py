"""An answer that fights its own reason is asked again, once.

LIVE 2026-09-28: "3 of 5, between 'makes lists' and 'relies on memory'. I am
choosing the middle option because I genuinely hold a strong preference for
externalized structure over relying on internal memory." A reason that names one
side and an answer that commits to neither is an answer contradicting itself,
and nothing noticed.

Measured from the page's own words rather than a vocabulary, and conservative:
where the reason names neither side or both equally it says nothing, because a
check that guesses is worse than no check. Nothing here rewrites what she chose.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def _row(count: int = 5, left: str = "makes lists", right: str = "relies on memory"):
    asks = f"{left} " + " ".join(f"[{n}]" for n in range(1, count + 1)) + f" {right}"
    return [
        {"group": "Q1", "role": "radio", "name": "Q1", "value": str(n),
         "selector": f"#Q1V{n}", "asks": asks}
        for n in range(1, count + 1)
    ]


def test_the_live_case_is_caught():
    said = S._the_choice_disagrees_with_its_reason(
        _row(), 2,
        "I am choosing the middle option because I genuinely hold a strong "
        "preference for externalized structure over relying on internal memory.",
    )
    assert "midpoint" in said
    assert "commits to neither" in said


def test_an_answer_on_the_wrong_side_is_caught():
    said = S._the_choice_disagrees_with_its_reason(
        _row(), 0, "I rely on memory constantly."
    )
    assert 'leans toward "makes lists"' in said


def test_an_answer_that_follows_its_reason_is_left_alone():
    assert S._the_choice_disagrees_with_its_reason(
        _row(), 4, "I rely on memory constantly."
    ) == ""
    assert S._the_choice_disagrees_with_its_reason(
        _row(), 0, "I make lists for everything."
    ) == ""


def test_a_reason_naming_neither_side_says_nothing():
    assert S._the_choice_disagrees_with_its_reason(
        _row(), 2, "I am genuinely balanced here."
    ) == ""


def test_a_reason_naming_both_equally_says_nothing():
    """A check that guesses is worse than no check."""
    assert S._the_choice_disagrees_with_its_reason(
        _row(), 2, "I make lists and I rely on memory in equal measure."
    ) == ""


def test_labelled_options_are_not_judged_this_way():
    labelled = [
        {"group": "q", "name": name, "asks": "how much? [] [] []"}
        for name in ("agree", "neutral", "disagree")
    ]
    assert S._the_choice_disagrees_with_its_reason(labelled, 1, "I agree") == ""


def test_she_is_asked_again_and_the_second_answer_stands(monkeypatch):
    skill = S()
    asked: list[str] = []
    answers = [
        {"actions": [{"index": 2}], "why": "I rely on memory constantly."},
        {"actions": [{"index": 4}], "why": "I rely on memory constantly."},
    ]

    async def _decide(goal, observation, history, understanding=None, **kwargs):
        asked.append(str(kwargs.get("noticed") or ""))
        return answers[min(len(asked) - 1, len(answers) - 1)]

    monkeypatch.setattr(skill, "_decide_next_actions", _decide)
    observation = {"url": "u", "title": "t", "text": "x", "elements": _row()}
    result = asyncio.run(
        skill._answer_each_question("take it", {**observation, "elements": _row() + _row_q2()}, [], None)
    )
    assert len(asked) >= 2, "a contradicting answer was not asked again"
    assert asked[0] == "", "the first ask must not carry a notice"
    assert "commits to neither" in asked[1]
    assert result and "#Q1V5" in str(result)


def _row_q2():
    asks = "sceptical [1] [2] [3] [4] [5] wants to believe"
    return [
        {"group": "Q2", "role": "radio", "name": "Q2", "value": str(n),
         "selector": f"#Q2V{n}", "asks": asks}
        for n in range(1, 6)
    ]


def test_the_notice_says_what_disagrees_and_not_what_to_pick(monkeypatch):
    """Telling her what to choose would make the answer the check's."""
    skill = S()
    notices: list[str] = []

    async def _decide(goal, observation, history, understanding=None, **kwargs):
        notices.append(str(kwargs.get("noticed") or ""))
        return {"actions": [{"index": 2}], "why": "I rely on memory constantly."}

    monkeypatch.setattr(skill, "_decide_next_actions", _decide)
    asyncio.run(
        skill._answer_each_question(
            "take it",
            {"url": "u", "title": "t", "text": "x", "elements": _row() + _row_q2()},
            [], None,
        )
    )
    said = [notice for notice in notices if notice]
    assert said
    for notice in said:
        lowered = notice.lower()
        assert "choose" not in lowered.replace("the answer chosen", "")
        assert "should" not in lowered


def test_a_graded_question_is_answered_in_two_steps():
    """Where she stands first; the position follows from it.

    Asked "on a scale of 1 to 10, 10 being love and 1 being hate, how much do
    you like chocolate", a person does not weigh ten dots — they know where they
    stand (it is their favourite; they are allergic; it is fine but not their
    first choice) and the number follows. Picking an index straight off has no
    stance behind it, and the safe-looking index is the middle.
    """
    assert "stand" in S._DECISION_SCHEMA["properties"]
    assert "stand" not in S._DECISION_SCHEMA["required"], (
        "a Next button has no position to take"
    )


def test_her_stance_is_what_the_position_is_checked_against():
    """The reason for the place and the place itself are different claims."""
    decision = {
        "actions": [{"index": 2}],
        "stand": "I externalise everything; I make lists for everything.",
        "why": "the middle felt safe",
    }
    said = S._first_disagreement(decision, _row())
    assert "midpoint" in said


def test_her_stance_is_said_before_the_reason_for_the_place():
    options = _row(left="sceptical", right="wants to believe")
    said = S._an_answer_in_words(
        options, 1, "I hold truth above comfort. so a 2 rather than a 1."
    )
    assert said.index("I hold truth above comfort") < said.index("so a 2")
