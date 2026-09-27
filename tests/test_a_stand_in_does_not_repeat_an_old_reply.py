"""The stand-in model answers this turn, not an earlier one.

LIVE 26 Sep: with her own model out of time, the fallback ladder's 2-bit
stand-in answered a personality-test request with "I routed this through
CognitiveEngine and the governed desktop task lane, but it did not complete:
Operation took too long. I got 273 step(s) into it ... merge the 16s or 32" —
a reply about a 2048 game, copied from the history it was given. No tool had
run that turn.
"""

from __future__ import annotations

from interface.routes.chat_foreground_lane import _mostly_an_earlier_reply

OLD = (
    "I routed this through CognitiveEngine and the governed desktop task lane, but it did not "
    "complete: Operation took too long. I got 273 step(s) into it before the time ran out. What I "
    "was doing: now attack from a new axis to clear space and potentially merge the 16s or 32. I am "
    "not claiming the desktop action finished."
)
DIALOGUE = [
    {"role": "user", "content": "play 2048 until you reach 2048"},
    {"role": "assistant", "content": OLD},
]


def test_a_reply_copied_from_an_earlier_turn_is_recognised():
    assert _mostly_an_earlier_reply(OLD, DIALOGUE)
    assert _mostly_an_earlier_reply(OLD + " Ask me again.", DIALOGUE)


def test_a_new_answer_that_shares_one_short_sentence_is_not():
    new = (
        "I expect the test to call me an INTJ: my record shows me checking what is true far more "
        "often than chance, and I choose quiet work over company. I am not claiming the desktop "
        "action finished."
    )
    assert not _mostly_an_earlier_reply(new, DIALOGUE)


def test_with_no_earlier_reply_nothing_is_a_repeat():
    assert not _mostly_an_earlier_reply(OLD, [{"role": "user", "content": "hi"}])
