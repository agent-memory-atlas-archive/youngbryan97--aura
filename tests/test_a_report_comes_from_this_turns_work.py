"""A turn reports the work it did, not the work that happened to be recorded.

LIVE 2026-09-27. Asked to take a personality test, the turn ran out of time
and the person was told:

    I got 273 step(s) into it before the time ran out. What I was doing: now
    attack from a new axis to clear space and potentially merge the 16s or 32.

That is a game of 2048 from an earlier turn. The reporter took whichever of
her two records had the most steps, which was the right rule for choosing
between two records of the same turn and the wrong rule for deciding which
turn a record belongs to.
"""

from __future__ import annotations

import time

import pytest

from core.agency import what_she_is_doing as doing


@pytest.fixture(autouse=True)
def _clean():
    doing._current = None
    doing._last = None
    yield
    doing._current = None
    doing._last = None


def _a_game_of_2048(steps: int) -> None:
    doing.taking_on("play 2048", where="a browser tab")
    doing.going_about_it("attack from a new axis to clear space", lived=False)
    for _ in range(steps):
        doing.a_step_taken()


def test_work_from_before_the_turn_is_not_this_turns_work():
    _a_game_of_2048(273)
    turn_began = time.time()
    assert doing.work_begun_since(turn_began) is None


def test_work_this_turn_began_is_reported():
    turn_began = time.time()
    doing.taking_on("take the personality test", where="a browser tab")
    doing.going_about_it("answer each item from my own record of choices", lived=False)
    for _ in range(4):
        doing.a_step_taken()
    held = doing.work_begun_since(turn_began)
    assert held is not None and held.steps == 4
    assert "personality test" in held.goal


def test_between_this_turns_records_the_one_that_did_the_work_wins():
    """The rule the max was written for: a record created after the work."""
    turn_began = time.time()
    doing.taking_on("take the personality test")
    doing.going_about_it("answer each item", lived=False)
    for _ in range(9):
        doing.a_step_taken()
    # Something else takes over the moment the work ends, pushing the record
    # of the work into `_last`.
    doing.taking_on("tidy up afterwards")
    held = doing.work_begun_since(turn_began)
    assert held is not None and held.steps == 9


def test_the_reporter_asks_only_for_this_turns_work():
    from interface.routes.chat_desktop_objective import _what_she_got_through

    _a_game_of_2048(273)
    turn_began = time.time()
    said = _what_she_got_through(0, 0, since=turn_began)
    assert "273" not in said
    assert said == "Completed 0/0 steps."

    doing.taking_on("take the personality test")
    doing.going_about_it("answer each item from my own record", lived=False)
    for _ in range(6):
        doing.a_step_taken()
    said = _what_she_got_through(0, 0, since=turn_began)
    assert "6 step(s)" in said
    assert "answer each item from my own record" in said


def test_a_receipt_still_outranks_her_record():
    from interface.routes.chat_desktop_objective import _what_she_got_through

    _a_game_of_2048(273)
    assert _what_she_got_through(2, 3, since=time.time()) == "Completed 2/3 steps."
