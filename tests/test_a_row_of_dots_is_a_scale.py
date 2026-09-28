"""A run of unlabelled controls between two phrases is a scale, not a list.

The page draws five dots between "makes lists" and "relies on memory" and says
nowhere that the position IS the answer. Shown as options they read as five
nameless things to pick from, and the middle one is the only safe pick — which
is what a watcher saw on 2026-09-28, "3 of 5" on item after item.

Structural, so it holds for any instrument that draws one: the layout the
observer already captures puts the option run between the words on either side
of it. Nothing here knows what a personality test is.
"""
from __future__ import annotations

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def _scale(count: int = 5, left: str = "makes lists", right: str = "relies on memory"):
    asks = f"{left} " + " ".join(f"[{n}]" for n in range(1, count + 1)) + f" {right}"
    return [
        {"group": "Q1", "name": "Q1", "value": str(n), "asks": asks}
        for n in range(1, count + 1)
    ]


def test_the_two_ends_are_read_from_the_layout():
    assert S._the_scale_it_offers(_scale()) == ("makes lists", "relies on memory", 5)


def test_what_each_position_means_is_said():
    reads = S._how_the_scale_reads(_scale())
    assert '1 is entirely "makes lists"' in reads
    assert '5 is entirely "relies on memory"' in reads
    assert "3 is the midpoint" in reads


def test_an_even_scale_has_no_midpoint_to_hide_in():
    reads = S._how_the_scale_reads(_scale(count=6))
    assert "no midpoint" in reads


def test_options_that_say_what_they_are_are_not_a_scale():
    labelled = [
        {"group": "q", "name": name, "asks": "x [] [] [] y"}
        for name in ("agree", "neutral", "disagree")
    ]
    assert S._the_scale_it_offers(labelled) is None
    assert S._how_the_scale_reads(labelled) == ""


def test_a_run_with_words_on_one_side_only_is_not_a_scale():
    one_sided = [
        {"group": "q", "name": "q", "value": str(n), "asks": f"pick one [1] [2] [3]"}
        for n in range(1, 4)
    ]
    assert S._the_scale_it_offers(one_sided) is None


def test_two_controls_are_a_choice_rather_than_a_scale():
    pair = [
        {"group": "q", "name": "q", "value": str(n), "asks": "yes [1] [2] no"}
        for n in range(1, 3)
    ]
    assert S._the_scale_it_offers(pair) is None


def test_the_page_she_is_shown_says_what_the_dots_are():
    observation = {
        "url": "https://example.test",
        "title": "t",
        "text": "x",
        "elements": [
            {**option, "role": "radio", "selector": f"#Q1V{n}"}
            for n, option in enumerate(_scale(), start=1)
        ],
    }
    rendered = S._render_observation(observation, "take the test")
    assert "5-point scale between two opposites" in rendered


def test_an_answer_says_which_way_it_leans():
    options = _scale(left="sceptical", right="wants to believe")
    assert 'toward "sceptical"' in S._an_answer_in_words(options, 0, "")
    assert 'toward "wants to believe"' in S._an_answer_in_words(options, 4, "")
    assert "midpoint" in S._an_answer_in_words(options, 2, "")


def test_a_labelled_answer_still_says_its_own_label():
    labelled = [
        {"group": "q", "name": name, "asks": f"how much? [] [] []"}
        for name in ("agree", "neutral", "disagree")
    ]
    said = S._an_answer_in_words(labelled, 0, "")
    assert "agree" in said
    assert "toward" not in said
