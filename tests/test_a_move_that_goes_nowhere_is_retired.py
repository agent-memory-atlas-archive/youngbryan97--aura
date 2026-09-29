"""A pair of controls that undo each other is not progress, and the stall counter cannot see it.

The loop already refuses to keep going when a round changes nothing: it compares
the page with the page before it and gives up after a few identical ones. A pair
of controls that undo each other defeats that exactly. Every round genuinely
changes the page, so the counter resets every round, and the page alternates
between two states for as long as the budget lasts.

LIVE 2026-09-29, on her own result page: `[more]` expanded a section and `[less]`
collapsed it, twenty-two of forty rounds went into the pair, and that was about
an hour of a seventy-minute run — spent on a page she had already read.

What is retired is a control that led back to a state this run has already been
in. Nothing here knows what a disclosure control is, or which page it is on.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def _page(*selectors: str) -> dict[str, object]:
    return {
        "url": "https://example.org/result",
        "elements": [
            {"role": "link", "name": selector.strip("#"), "selector": selector}
            for selector in selectors
        ],
    }


def test_a_retired_control_is_gone_from_the_page_everything_reads():
    page = _page("#more", "#less", "#done")
    left = S._without_the_moves_that_go_nowhere(page, {"#less"})
    assert [element["selector"] for element in left["elements"]] == ["#more", "#done"]
    # The observation it was given is untouched: two lists built from one page is
    # how a loop comes to name one control and press another.
    assert len(page["elements"]) == 3


def test_retiring_nothing_returns_the_same_page():
    page = _page("#more", "#less")
    assert S._without_the_moves_that_go_nowhere(page, set()) is page


def test_a_page_with_nothing_left_is_returned_whole():
    """Retiring is for choosing between ways on, never the reason there is none."""
    page = _page("#more", "#less")
    left = S._without_the_moves_that_go_nowhere(page, {"#more", "#less"})
    assert [element["selector"] for element in left["elements"]] == ["#more", "#less"]


def test_the_two_states_of_a_toggle_have_different_signatures():
    """Which is why the stall counter never fired: each round did change the page."""
    expanded = {"url": "u", "elements": [{"role": "link", "name": "less", "checked": None}]}
    collapsed = {"url": "u", "elements": [{"role": "link", "name": "more", "checked": None}]}
    assert S._observation_signature(expanded) != S._observation_signature(collapsed)
    # And why being-here-before is the fact worth acting on: the run returns to a
    # state it has already been in, which is what a set of them records.
    been_in = {S._observation_signature(expanded), S._observation_signature(collapsed)}
    assert S._observation_signature(expanded) in been_in
