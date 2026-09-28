"""A graded answer is a measurement of her against a described dimension.

Handing the dimension to a language model and taking the number it writes is not
her answering: the model has no access to what she has valued or chosen, so it
writes a plausible sentence and picks the position that commits to nothing.
Measured live on 2026-09-28, that position was the midpoint on item after item,
under reasons that named a strong preference.

Her record already holds what she is like. Each side of a dimension either
matches it or does not, the difference is a lean, and a lean maps onto the
positions a page offers. Nothing here knows what any instrument measures.
"""
from __future__ import annotations

from typing import Any

import pytest

from core.self import where_i_stand as w

pytestmark = pytest.mark.unit


class _Embedder:
    """Vectors over a tiny vocabulary, so a match is arithmetic and not luck."""

    _WORDS = ("truth", "care", "order", "memory", "play")

    def embed(self, text: str) -> list[float]:
        said = str(text or "").lower()
        return [1.0 if word in said else 0.0 for word in self._WORDS]


@pytest.fixture
def embedder(monkeypatch):
    engine = _Embedder()
    monkeypatch.setattr(w, "_embedder", lambda: engine)
    return engine


def _record() -> list[w.Piece]:
    return [
        w.Piece(said="truth", weight=1.05, source="value"),
        w.Piece(said="care", weight=1.04, source="value"),
        w.Piece(said="order", weight=0.9, source="chose"),
    ]


def test_a_side_her_record_names_is_more_her(embedder):
    match, because = w.how_much_it_is_her("truth and order", _record())
    assert match > 0.0
    assert because


def test_a_side_nothing_in_her_record_names_is_less_her(embedder):
    near, _ = w.how_much_it_is_her("truth", _record())
    far, _ = w.how_much_it_is_her("play", _record())
    assert near > far


def test_the_lean_points_at_the_side_that_is_more_her(embedder):
    lean = w.where_she_stands("play", "truth", _record())
    assert lean.measured
    assert lean.toward > 0.0, "the second side is the one her record names"
    flipped = w.where_she_stands("truth", "play", _record())
    assert flipped.toward < 0.0


def test_the_lean_is_symmetric_under_swapping_the_sides(embedder):
    one = w.where_she_stands("play", "truth", _record())
    other = w.where_she_stands("truth", "play", _record())
    assert one.toward == pytest.approx(-other.toward, abs=1e-9)


def test_two_sides_her_record_treats_alike_lean_neither_way(embedder):
    lean = w.where_she_stands("truth", "truth", _record())
    assert lean.measured
    assert lean.toward == pytest.approx(0.0, abs=1e-9)


def test_a_lean_names_a_position_on_any_length_of_run(embedder):
    lean = w.where_she_stands("play", "truth", _record())
    for count in (2, 3, 5, 7):
        place = lean.position_in(count)
        assert place is not None
        assert 0 <= place < count
    assert w.Lean(toward=-1.0, first=0, second=0, measured=True).position_in(5) == 0
    assert w.Lean(toward=1.0, first=0, second=0, measured=True).position_in(5) == 4
    assert w.Lean(toward=0.0, first=0, second=0, measured=True).position_in(5) == 2


def test_an_empty_record_measures_nothing_rather_than_the_middle():
    """No evidence and "equally both" are different answers."""
    lean = w.where_she_stands("one thing", "another", [])
    assert lean.measured is False
    assert lean.position_in(5) is None


def test_no_embedder_measures_nothing(monkeypatch):
    monkeypatch.setattr(w, "_embedder", lambda: None)
    lean = w.where_she_stands("one thing", "another", _record())
    assert lean.measured is False


def test_the_measurement_says_what_matched(embedder):
    lean = w.where_she_stands("play", "truth and care", _record())
    assert lean.because, "a number nobody can argue with is not evidence"
    assert any("truth" in said or "care" in said for said in lean.because)


def test_her_record_is_made_of_things_she_produced():
    """Values she holds, choices she made, words she said. Nothing invented."""
    import inspect

    body = inspect.getsource(w.her_record)
    assert "what_she_is_like" in body
    assert "stated_preferences" in body
    assert "source=\"value\"" in body and "source=\"chose\"" in body


def test_nothing_here_knows_what_an_instrument_measures():
    import inspect

    source = inspect.getsource(w)
    for tuned in ("jungian", "extravert", "introvert", "oejts", "myers", "likert"):
        assert tuned not in source.lower(), f"the measure is tuned to {tuned!r}"
