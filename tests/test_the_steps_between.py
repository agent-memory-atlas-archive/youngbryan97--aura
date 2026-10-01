"""Gaps between amounts are counted in the steps the world takes, not in doublings.

Her judgement of a position counted nearness and smoothness in doublings,
because "in anything built by combining a step is a doubling" — true of 2048
and of little else. The step is now read off the amounts a world shows.
"""
from __future__ import annotations

import math

import pytest

from core.agency.how_good_is_this import _smoothness
from core.agency.the_steps_between import in_view, ladder_here, ladder_of
from core.agency.what_she_is_after import goal_in
from core.perception.what_is_there import arranged


def board(rows):
    return arranged([
        (0.2 + r * 0.15, 0.2 + c * 0.15, said)
        for r, row in enumerate(rows)
        for c, said in enumerate(row)
        if said
    ])


def test_a_world_that_doubles_is_measured_in_doublings() -> None:
    ladder = ladder_of([2, 4, 8, 16])
    assert ladder.multiplies
    assert ladder.place(2048) == pytest.approx(11.0)


def test_a_world_that_adds_one_is_measured_in_ones() -> None:
    ladder = ladder_of([1, 2, 3, 4, 5])
    assert not ladder.multiplies
    assert ladder.place(5) == pytest.approx(5.0)
    assert ladder.apart(2, 5) == pytest.approx(3.0)


def test_neighbours_one_apart_in_an_adding_world_are_as_close_as_neighbours_can_be() -> None:
    adding = board([["1", "2", "3", "4"]])
    doubling = board([["2", "4", "8", "16"]])
    assert _smoothness(adding) == pytest.approx(_smoothness(doubling)) == pytest.approx(0.5)


def test_nearness_to_a_goal_follows_the_worlds_step() -> None:
    adding = board([["1", "2", "3", "4", "5", "6"]])
    assert goal_in("12").nearness(adding) == pytest.approx(6 / 12)
    doubling = board([["2", "4", "8", "64"]])
    assert goal_in("2048").nearness(doubling) == pytest.approx(math.log2(64) / math.log2(2048))


def test_a_search_measures_by_every_amount_it_has_seen() -> None:
    with in_view():
        ladder_here([2, 8])
        ladder = ladder_here([4])
    assert ladder.place(8) == pytest.approx(3.0)
    assert ladder_here([4]) is None
