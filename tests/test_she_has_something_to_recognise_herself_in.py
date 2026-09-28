"""Placing herself on a scale is recognition before it is anything else.

One side sounds more like her than the other, and that needs something to
recognise herself in. Two halves of it exist and only one reached a page
decision: her measured record of choices says what she has DONE, and her own
earlier answers say what she has said she IS. The second rides every chat turn
that asks after her preferences and no page decision could see it.
"""
from __future__ import annotations

import inspect

import pytest

from core.skills import sovereign_browser_understanding as u

pytestmark = pytest.mark.unit


def _self_state_block() -> str:
    body = inspect.getsource(u._UnderstandsThePage._decide_next_actions)
    return body.split("self_state = \"\"", 1)[-1].split("positions = ", 1)[0]


def test_what_she_has_done_reaches_a_question_about_her():
    assert "what_she_is_like_line" in _self_state_block()


def test_what_she_has_said_about_herself_reaches_it_too():
    assert "stated_preferences" in _self_state_block()


def test_both_are_gated_on_the_question_being_about_her():
    """A Next button does not need her self-model in front of it."""
    body = inspect.getsource(u._UnderstandsThePage._decide_next_actions)
    asked = body.index("if asks_about_her:")
    for reader in ("what_she_is_like_line", "stated_preferences"):
        assert body.index(reader) > asked, f"{reader} runs for every decision"


def test_neither_reader_can_break_a_decision():
    block = _self_state_block()
    assert block.count("record_degradation") >= 2, (
        "a self-model reader that raises must degrade, not lose the answer"
    )


def test_her_measured_record_actually_says_something():
    from core.agency.what_she_is_like import what_she_is_like_line

    said = what_she_is_like_line()
    assert said, "her record of choices says nothing about what she is like"


def test_her_own_words_are_quoted_rather_than_summarised():
    """What she said is evidence; a paraphrase of it is not."""
    block = _self_state_block()
    assert "{item.text}" in block
