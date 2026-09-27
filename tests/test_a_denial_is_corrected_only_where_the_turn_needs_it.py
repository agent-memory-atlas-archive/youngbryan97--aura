"""A denial is corrected only against what this turn needs.

LIVE 27 Sep, in a reply about taking a personality test on a named site, her
sentence "I can't actually take the test in this turn" was replaced with "I do
have diagnose repo — it is registered and enabled right now". The word "test"
named the repository diagnoser, and the set of capabilities the turn needed
was empty, so every denial counted as on topic.
"""

from __future__ import annotations

import contextvars

from interface.routes.chat_lane_bookkeeping import _capabilities_this_turn_needs

REQUEST = (
    "Take the Open Extended Jungian Type Scales personality test on openpsychometrics.org. "
    "Before you start, tell me what type you think it will give you and why."
)


def _in_a_turn(question, call):
    from core.conversation.session_scope import set_user_question

    def run():
        set_user_question(question)
        return call()

    return contextvars.copy_context().run(run)


def test_page_work_needs_the_browser_that_does_it():
    needs = _in_a_turn(REQUEST, _capabilities_this_turn_needs)
    assert {"sovereign_browser", "desktop_task"} <= needs


def test_a_turn_that_is_not_page_work_is_not_given_the_browser():
    needs = _in_a_turn("what is the Myers-Briggs test?", _capabilities_this_turn_needs)
    assert "sovereign_browser" not in needs


def test_a_personality_test_is_not_corrected_into_the_repository_diagnoser():
    from interface.routes.chat_reply_shaping import _correct_false_capability_denials

    said = "I can't actually take the test in this turn. I expect INTP."
    corrected = _in_a_turn(REQUEST, lambda: str(_correct_false_capability_denials(said)))
    assert "diagnose repo" not in corrected
    assert "I expect INTP." in corrected
