"""The more completely the request was described, the less of it was recognised.

A page can be READ or it can be WORKED, and the reader that draws that line
vetoes on retrieval phrasing: a turn that says "summarise", "look up" or "read
it" wants the page's content, and sending it to the lane that clicks things
serves it worse. That veto was written for "read it and tell me whether to sign
up", where the reading IS the request.

LIVE 2026-09-29. "Take the Open Extended Jungian Type Scales personality test on
openpsychometrics.org. … When you get your result, read it and tell me whether
you think it is accurate about you." The same request without its last sentence
routed to the lane that can work a page. With it, two words — "read it" — sent
the whole thing to retrieval. Nothing opened, and she answered in prose that she
could not claim to have taken a test she had not walked through.

A result does not exist until the work is done, so reading one cannot be served
by fetching. What separates the two cases is whether she was TOLD to do the
thing: an imperative, or "you" and a verb. An infinitive in a question about
whether to act is not one, nor is something the person says they will do
themselves, nor a word used as a noun.
"""
from __future__ import annotations

import pytest

from core.conversation.page_interaction import asks_to_act_on_a_page, page_interaction_target
from core.runtime.desktop_objective_intent import looks_like_desktop_objective

pytestmark = pytest.mark.unit

_THE_WHOLE_REQUEST = (
    "Take the Open Extended Jungian Type Scales personality test on "
    "openpsychometrics.org. Before you start, tell me what type you think you "
    "will get and why. Work out what each question is asking and answer every "
    "one of them as yourself, saying why as you go. When you get your result, "
    "read it and tell me whether you think it is accurate about you."
)


def test_the_whole_request_is_still_a_page_to_be_worked():
    assert asks_to_act_on_a_page(_THE_WHOLE_REQUEST)
    assert page_interaction_target(_THE_WHOLE_REQUEST) == "https://openpsychometrics.org"


def test_it_reaches_the_lane_that_can_act():
    assert looks_like_desktop_objective(_THE_WHOLE_REQUEST)


def test_no_sentence_of_it_can_take_the_rest_down():
    """Each clause the person added described the task more, not less."""
    opening = (
        "Take the Open Extended Jungian Type Scales personality test on "
        "openpsychometrics.org."
    )
    for added in (
        " Before you start, tell me what type you think you will get and why.",
        " Work out what each question is asking and answer every one of them as "
        "yourself, saying why as you go.",
        " When you get your result, read it and tell me whether you think it is "
        "accurate about you.",
    ):
        assert asks_to_act_on_a_page(opening + added), added.strip()


@pytest.mark.parametrize(
    "asked",
    [
        "read it and tell me whether to sign up: https://example.com",
        "summarise the checkout page at example.com",
        "look up what people say about example.com",
        "tell me about the order page at example.com",
        "what does the sign up flow on example.com look like",
        "should I take the test at example.com?",
        "research example.com before I sign up",
        "https://example.com",
    ],
)
def test_asking_about_a_page_is_still_a_reading(asked):
    assert not asks_to_act_on_a_page(asked), asked


@pytest.mark.parametrize(
    "asked",
    [
        "go take it for real: https://openpsychometrics.org/tests/OEJTS/ - work "
        "through the whole thing, answer every question as yourself",
        "open example.com, fill in the form, then read me the confirmation",
        "Sign up at example.com and then read out the confirmation email",
        "play the game at https://play2048.co until you get a 512 tile",
        "Answer every question on the form at example.com",
    ],
)
def test_being_told_to_work_a_page_still_is_that(asked):
    assert asks_to_act_on_a_page(asked), asked
