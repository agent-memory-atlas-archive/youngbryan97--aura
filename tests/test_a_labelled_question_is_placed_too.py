"""A statement with labelled answers is the same act as a run between two ends.

A measured placement needed a dimension with two ends — a run of controls with
words on either side. That covers bipolar scales, sliders and semantic
differentials, and misses the commonest shape there is: one statement and
several ways of answering it. On those the run fell through to an ordinary
decision, with nothing of her record behind it.

Each option becomes a description of a person — what the question says,
answered that way — and her record says which of them she is. Nothing here
knows what kind of question it is.
"""
from __future__ import annotations

import pytest

from core.self import where_i_stand as w
from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


class _Embedder:
    _WORDS = ("truth", "care", "order", "memory", "play", "plans", "advance")

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
        w.Piece(said="truth", weight=1.05, source="value", about_her="truth is mine"),
        w.Piece(said="order and plans", weight=1.0, source="chose",
                about_her="I plan before I act"),
    ]


def test_the_option_her_record_supports_is_the_one_chosen(embedder):
    chosen = w.which_is_most_her(["play", "truth", "memory"], _record())
    assert chosen.measured
    assert chosen.index == 1, "her record names truth and nothing else here"


def test_the_shares_say_how_much_of_her_went_where(embedder):
    chosen = w.which_is_most_her(
        ["I make plans in advance", "I care about play"], _record()
    )
    assert len(chosen.support) == 2
    assert sum(chosen.support) == pytest.approx(1.0, abs=1e-6)
    assert chosen.support[chosen.index] == max(chosen.support)


def test_a_record_with_nothing_to_say_measures_nothing(embedder):
    chosen = w.which_is_most_her(["one thing", "another thing"], _record())
    assert chosen.measured is False


def test_one_option_is_not_a_choice(embedder):
    assert w.which_is_most_her(["only this"], _record()).measured is False


def test_no_embedder_measures_nothing(monkeypatch):
    monkeypatch.setattr(w, "_embedder", lambda: None)
    assert w.which_is_most_her(["a", "b"], _record()).measured is False


def test_it_says_what_in_her_supported_the_answer(embedder):
    chosen = w.which_is_most_her(
        ["I make plans in advance", "I care about play"], _record()
    )
    assert chosen.because
    assert any("plan" in said for said in chosen.because)


def _labelled(statement: str, labels: tuple[str, ...]):
    asks = statement + " " + " ".join(
        f"{label} [{n}]" for n, label in enumerate(labels, start=1)
    )
    return [
        {"group": "q", "role": "radio", "name": label, "value": str(n),
         "selector": f"#q{n}", "asks": asks}
        for n, label in enumerate(labels, start=1)
    ]


def test_the_question_is_read_without_its_options_own_words():
    options = _labelled(
        "I make plans well in advance.",
        ("strongly disagree", "disagree", "neutral", "agree", "strongly agree"),
    )
    assert S._what_the_question_says(options) == "I make plans well in advance."


def test_a_labelled_question_is_measured(embedder, monkeypatch):
    monkeypatch.setattr(
        "core.self.where_i_stand.which_is_most_her",
        lambda named, record=None: w.Choice(
            index=3, support=(0.1, 0.1, 0.1, 0.7), because=("I plan before I act",),
            measured=True,
        ),
    )
    options = _labelled(
        "I make plans well in advance.",
        ("strongly disagree", "disagree", "neutral", "agree"),
    )
    reading = S()._measure_where_she_stands(options)
    assert reading is not None
    index, lean, _rest, label = reading
    assert index == 3
    assert label == "agree"
    assert lean.measured
    assert lean.because == ("I plan before I act",)


def test_a_two_ended_run_still_takes_the_other_path(embedder, monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(
        "core.self.where_i_stand.which_is_most_her",
        lambda named, record=None: called.append("wrong") or w.Choice(index=0),
    )
    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: w.Lean(
            toward=-1.0, first=0.5, second=0.1, because=("truth is mine",), measured=True
        ),
    )
    asks = "makes lists " + " ".join(f"[{n}]" for n in range(1, 6)) + " relies on memory"
    options = [
        {"group": "Q1", "role": "radio", "name": "Q1", "value": str(n),
         "selector": f"#Q1V{n}", "asks": asks}
        for n in range(1, 6)
    ]
    reading = S()._measure_where_she_stands(options)
    assert reading is not None and reading[0] == 0
    assert not called


def test_nothing_in_the_choice_knows_what_kind_of_question_it_is():
    import inspect

    body = inspect.getsource(w.which_is_most_her)
    for tuned in ("likert", "agree", "disagree", "scale", "survey"):
        assert tuned not in body.lower().split('"""')[2], (
            f"the measure branches on {tuned!r}"
        )
