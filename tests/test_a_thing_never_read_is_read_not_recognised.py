"""A look near one thing she knows may be a thing she has never read.

Measured on the 2048 app, 26 Sep: a 256 sits 8.4 Lab units from a 128, inside
the 9.0 one look was allowed, while the same number at two places differs by
at most 2.5. With only 128s learned, the first 256 of a game was recognised
as a 128, and her rule for the game, which had it right, was scored as wrong:
"her rule missed on left: (2,1) said '256' saw '128'".
"""
from __future__ import annotations

import numpy as np

from core.perception.what_the_pixels_show import Looker


def _a_look(shade: float) -> np.ndarray:
    """A small picture every sample of which sits ``shade`` Lab units from grey 50."""
    look = np.full((24, 24, 3), 50.0)
    look[..., 0] += shade
    return look


def test_a_look_as_far_from_a_thing_as_a_new_thing_sits_is_read():
    looker = Looker()
    looker.learned(_a_look(0.0), "128")
    looker.learned(_a_look(2.5), "128")  # the same number at another place
    assert looker.recognised(_a_look(1.0)) == "128"
    assert looker.recognised(_a_look(8.4)) is None, "a 256 is not a 128 because nothing else is nearer"


def test_nothing_is_recognised_but_its_own_look_before_one_thing_is_read_twice():
    looker = Looker()
    looker.learned(_a_look(0.0), "2")
    assert looker.how_far_one_thing_varies() == 0.0
    assert looker.recognised(_a_look(0.0)) == "2"
    assert looker.recognised(_a_look(1.0)) is None


def test_how_far_one_thing_varies_is_learned_from_every_thing():
    looker = Looker()
    looker.learned(_a_look(0.0), "2")
    looker.learned(_a_look(1.2), "2")
    looker.learned(_a_look(30.0), "32")
    looker.learned(_a_look(32.5), "32")
    assert looker.how_far_one_thing_varies() == 2.5
    # A 2 seen as far from itself as a 32 has been is still a 2.
    assert looker.recognised(_a_look(2.0)) == "2"


def _a_board_with_an_eight(colour):
    from tests.test_the_places_are_seen_not_inferred import _a_board

    board = _a_board({(0, 0): colour, (2, 1): (218, 228, 238)})
    x, y = 47 + 48, 117 + 48
    board[y - 14 : y + 14, x - 9 : x + 9] = (250, 250, 250)
    return board


def test_a_place_reading_gives_nothing_for_is_the_one_thing_it_looks_near(monkeypatch):
    """The strip reads first; the nearest look answers only when it cannot.

    Holding recognition to the measured spread sent more places to the strip,
    and a strip reading one lone digit often returns nothing. Places that would
    not read went from once in 102 moves to 160 times in 1649 (live, 26 Sep).
    """
    from core.perception import what_the_pixels_show as pixels
    from core.perception.what_the_pixels_show import grids_in, panels_in
    from tests.test_the_places_are_seen_not_inferred import _words_at

    looker = Looker()
    first = _a_board_with_an_eight((121, 177, 242))
    grid = grids_in(panels_in(first))[0]
    looker.read(first, words=_words_at(grid, {(0, 0): "8", (2, 1): "2"}))
    learned = len(looker.seen)

    # The same tile a little differently drawn, and nothing reads.
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(Looker, "_read_as_a_strip", lambda self, image, grid_, spots: {})
    drifted = _a_board_with_an_eight((125, 180, 242))
    look = looker._look_of(drifted, grid.place(0, 0))
    assert looker.recognised(look) is None, "farther than one thing has been seen to vary"
    reading = looker.read(drifted)
    says = reading["grids"][0]["says"]
    assert says[0] == "8"
    assert [0, 0] not in reading["grids"][0]["unsure"]
    assert len(looker.seen) == learned, "a guess is not learned from"


def test_what_the_strip_reads_outranks_the_nearest_look(monkeypatch):
    """A 256 near a learned 128 is read as what the strip says, not as the 128."""
    from core.perception import what_the_pixels_show as pixels
    from core.perception.what_the_pixels_show import grids_in, panels_in
    from tests.test_the_places_are_seen_not_inferred import _words_at

    looker = Looker()
    first = _a_board_with_an_eight((121, 177, 242))
    grid = grids_in(panels_in(first))[0]
    looker.read(first, words=_words_at(grid, {(0, 0): "128", (2, 1): "2"}))
    monkeypatch.setattr(pixels, "recognize_text", lambda image: [])
    monkeypatch.setattr(
        Looker, "_read_as_a_strip", lambda self, image, grid_, spots: {spot: "256" for spot in spots if spot == (0, 0)}
    )
    reading = looker.read(_a_board_with_an_eight((125, 180, 242)))
    assert reading["grids"][0]["says"][0] == "256"
