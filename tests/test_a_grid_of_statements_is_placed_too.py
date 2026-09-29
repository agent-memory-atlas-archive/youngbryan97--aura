"""The commonest questionnaire on the web is a grid, and her record could place none of it.

A dimension names two things and asks which is more her. A statement with
labelled answers names one thing and several ways of answering it. Between them
those cover most of what a page can ask — and they miss the shape used more than
either: a column of statements with one scale printed above all of them.

A row of that grid has no second side to be weighed against, and which end of
the run means yes belongs to the heading rather than to the row. So both readers
returned nothing, and LIVE 2026-09-29 twenty-eight of a sixty-item instrument's
questions were answered by naming controls instead of by her: out of the page's
order, one of them five times over, and each with the same borrowed sentence.

Two measurements make it one question she can answer. How much her record bears
each statement out, with the rest of the grid supplying the contrast, and which
end of the run means yes, read off the two ends against each statement's own
affirmation and denial. Neither consults a list of agreeing words, because a
list like that is right on the page it was written for and backwards on the next.
"""
from __future__ import annotations

import re

import pytest

from core.self import where_i_stand as w
from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


class _Embedder:
    """A toy language: tokens as dimensions, with the two that carry polarity.

    Whole tokens, not substrings, because "disagree" contains "agree". And the
    orientation reading is a semantic one — it asks whether an end of a scale
    asserts a statement or denies it — so this fixture has to hold that much
    meaning or it is testing nothing: "agree" and "true" mean the same thing
    here, and so do "disagree" and "not".
    """

    _MEANS = {
        "agree": "affirm",
        "true": "affirm",
        "disagree": "deny",
        "not": "deny",
        "untrue": "deny",
    }
    _WORDS = (
        "truth", "care", "order", "lists", "memory", "crowds", "alone",
        "affirm", "deny",
    )

    def embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z]+", str(text or "").lower())
        held = {self._MEANS.get(token, token) for token in tokens}
        return [1.0 if word in held else 0.0 for word in self._WORDS]


@pytest.fixture
def embedder(monkeypatch):
    engine = _Embedder()
    monkeypatch.setattr(w, "_embedder", lambda: engine)
    return engine


def _record() -> list[w.Piece]:
    return [
        w.Piece(said="truth", weight=1.2, source="value", about_her="truth is mine"),
        w.Piece(said="order and lists", weight=1.0, source="chose",
                about_her="I keep lists"),
        w.Piece(said="care", weight=1.0, source="value", about_her="care is mine"),
    ]


def _grid(statement: str, columns: tuple[str, ...]) -> list[dict[str, object]]:
    """One row of a grid: five unlabelled radios under a labelled heading."""
    return [
        {
            "role": "radio",
            "group": "S1",
            "name": "S1",
            "selector": f"#S1V{place + 1}",
            "value": str(place + 1),
            "asks": f"{statement} " + " ".join(f"[{n + 1}]" for n in range(len(columns))),
            "column": column,
        }
        for place, column in enumerate(columns)
    ]


_COLUMNS = ("Disagree", "", "Neutral", "", "Agree")


def test_a_statement_under_a_heading_is_read_as_a_named_scale(embedder):
    options = _grid("I keep lists of what I have to do.", _COLUMNS)
    laid_out = S._how_the_options_are_laid_out(options)
    assert "a statement" in laid_out, laid_out
    assert '"Disagree" to "Agree"' in laid_out, laid_out
    shape = S._a_statement_on_a_named_scale(options)
    assert shape == ("I keep lists of what I have to do.", "Disagree", "Agree")


def test_a_run_whose_ends_carry_no_words_is_not_a_named_scale(embedder):
    # Only the middle column is labelled: the page has not named the ends, and
    # a scale invented from one word in the middle would be a rule of ours.
    options = _grid("I keep lists.", ("", "", "Neutral", "", ""))
    assert S._a_statement_on_a_named_scale(options) is None


#: A grid her record has something to say about, most supported first.
_STATEMENTS = (
    "I keep order and lists and care about truth",
    "I care about crowds",
    "I would rather have memory than order",
)


def test_her_record_places_every_statement_of_a_grid(embedder):
    leans = w.where_she_stands_on_a_grid(_STATEMENTS, "Disagree", "Agree", _record())
    assert all(lean.measured for lean in leans), "a grid her record cannot read at all"
    at = [lean.position_in(5) for lean in leans]
    assert len(set(at)) > 1, f"a grid answered identically throughout: {at}"
    # Placed in the order her record supports them, which is what a person does
    # with a page of statements: some ring truer than the rest of them.
    assert at == sorted(at, reverse=True), at


def test_a_statement_her_record_cannot_speak_to_is_left_unplaced(embedder):
    """No evidence is not "equally both", and the midpoint would say it was."""
    leans = w.where_she_stands_on_a_grid(
        ["I am at home in crowds", "I would rather be alone"], "Disagree", "Agree",
        _record(),
    )
    assert not any(lean.measured for lean in leans)
    assert all(lean.position_in(5) is None for lean in leans)


def test_a_scale_printed_backwards_puts_her_at_the_other_end(embedder):
    forward = w.where_she_stands_on_a_grid(_STATEMENTS, "Disagree", "Agree", _record())
    backward = w.where_she_stands_on_a_grid(_STATEMENTS, "Agree", "Disagree", _record())
    assert [lean.position_in(5) for lean in forward] == [
        4 - place for place in (lean.position_in(5) for lean in backward)
    ], "she stands where she stands; only the run is printed the other way"


def test_a_grid_lean_is_not_rescaled_a_second_time(embedder):
    """`against_the_rest` is for leans measured on their own, which saturate.

    A grid's leans are measured against the rest of the grid to begin with.
    Rescaling them by the widest of them a second time undid the answer: LIVE
    2026-09-29, her twenty-eight placements spread 5/7/6/2/8 across the five
    positions, and rescaled they were 2 at one end and 26 on the midpoint.
    """
    leans = w.where_she_stands_on_a_grid(_STATEMENTS, "Disagree", "Agree", _record())
    assert all(lean.relative for lean in leans)
    assert w.against_the_rest(leans) == pytest.approx([lean.toward for lean in leans])


def test_an_unoriented_run_places_nothing(embedder, monkeypatch):
    monkeypatch.setattr(w, "which_end_means_yes", lambda *a, **k: 0.0)
    leans = w.where_she_stands_on_a_grid(_STATEMENTS, "?", "?", _record())
    assert not any(lean.measured for lean in leans), (
        "a run nobody can orient would put every answer at the wrong end with "
        "the same confidence"
    )


def test_the_scales_own_word_is_what_she_says_she_answered(embedder):
    options = _grid("I keep lists of what I have to do.", _COLUMNS)
    said = S._an_answer_in_words(options, 4, "lists are how I hold a day together")
    assert "Agree" in said, said
    # A dot with no word of its own is said by the two it sits between.
    between = S._an_answer_in_words(options, 3, "")
    assert 'between "Neutral" and "Agree"' in between, between
