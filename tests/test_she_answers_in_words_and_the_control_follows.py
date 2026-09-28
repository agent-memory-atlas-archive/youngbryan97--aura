"""Her position is measured from her own record; the model says what it means.

The model is a linguistic organ and a semantic one. Asked to CHOOSE, it has no
access to what she has valued, chosen or said about herself, so it writes a
plausible sentence and takes the position that commits to nothing — the
midpoint, item after item, measured live on 2026-09-28.

The two things a question names come from the page. How much each is her comes
from her record. The difference is a lean, the lean names a position, and the
model is handed the measurement afterwards to say what it means.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.self.where_i_stand import Lean
from core.skills import sovereign_browser_understanding as u
from core.skills.sovereign_browser import SovereignBrowserSkill as S

pytestmark = pytest.mark.unit


def _row(count: int = 5, left: str = "makes lists", right: str = "relies on memory"):
    asks = f"{left} " + " ".join(f"[{n}]" for n in range(1, count + 1)) + f" {right}"
    return [
        {"group": "Q1", "role": "radio", "name": "Q1", "value": str(n),
         "selector": f"#Q1V{n}", "asks": asks}
        for n in range(1, count + 1)
    ]


def _skill(monkeypatch, lean: Lean, said: str = "because of what I value."):
    skill = S()
    handed: dict[str, Any] = {}

    def _stands(first: str, second: str, record=None) -> Lean:
        handed["sides"] = (first, second)
        return lean

    async def _asked(prompt: str, mind: str = "", *, shaped: bool = True):
        handed["prompt"] = prompt
        return said, "Cortex"

    async def _mind() -> str:
        return "her mind"

    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands", _stands, raising=True
    )
    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    monkeypatch.setattr(skill, "_assembled_mind", _mind)
    return skill, handed


def _run(skill, options):
    return asyncio.run(
        skill._where_she_puts_herself(
            "take it",
            {"url": "u", "title": "t", "text": "x", "elements": options},
            options,
            None,
        )
    )


def test_the_two_sides_are_read_from_the_page(monkeypatch):
    lean = Lean(toward=-0.9, first=0.4, second=0.1, because=("truth (value)",), measured=True)
    skill, handed = _skill(monkeypatch, lean)
    _run(skill, _row())
    assert handed["sides"] == ("makes lists", "relies on memory")


def test_a_lean_toward_the_first_side_takes_a_position_near_it(monkeypatch):
    lean = Lean(toward=-1.0, first=0.5, second=0.0, because=("truth (value)",), measured=True)
    skill, _handed = _skill(monkeypatch, lean)
    placed = _run(skill, _row())
    assert placed["selector"] == "#Q1V1"


def test_a_lean_toward_the_second_side_takes_a_position_near_it(monkeypatch):
    lean = Lean(toward=1.0, first=0.0, second=0.5, because=("care (value)",), measured=True)
    skill, _handed = _skill(monkeypatch, lean)
    placed = _run(skill, _row())
    assert placed["selector"] == "#Q1V5"


def test_the_midpoint_is_taken_only_when_her_record_is_even(monkeypatch):
    lean = Lean(toward=0.0, first=0.3, second=0.3, because=("truth (value)",), measured=True)
    skill, _handed = _skill(monkeypatch, lean)
    placed = _run(skill, _row())
    assert placed["selector"] == "#Q1V3"


def test_an_unmeasured_record_answers_nothing(monkeypatch):
    lean = Lean(toward=0.0, first=0.0, second=0.0, because=(), measured=False)
    skill, _handed = _skill(monkeypatch, lean)
    assert _run(skill, _row()) is None, "a guess must not be passed off as a measurement"


def test_the_model_is_handed_the_measurement_not_the_choice(monkeypatch):
    lean = Lean(toward=-0.8, first=0.44, second=0.12, because=("truth (value, +0.44)",), measured=True)
    skill, handed = _skill(monkeypatch, lean)
    _run(skill, _row())
    prompt = handed["prompt"]
    assert "+0.44" in prompt and "+0.12" in prompt
    assert "position 1 of 5" in prompt or "position 1" in prompt
    assert "already where you are" in prompt, (
        "the model must be saying what the position means, not picking one"
    )


def test_her_words_become_the_reason_that_is_narrated(monkeypatch):
    lean = Lean(toward=-0.8, first=0.4, second=0.1, because=("truth (value)",), measured=True)
    skill, _handed = _skill(monkeypatch, lean, said="Lists are how I hold truth steady.")
    placed = _run(skill, _row())
    assert "Lists are how I hold truth steady." in placed["said"]


def test_a_silent_model_does_not_lose_the_measured_answer(monkeypatch):
    lean = Lean(toward=-0.8, first=0.4, second=0.1, because=("truth (value)",), measured=True)
    skill = S()

    async def _asked(prompt: str, mind: str = "", *, shaped: bool = True):
        return "", ""

    async def _mind() -> str:
        return "her mind"

    monkeypatch.setattr(
        "core.self.where_i_stand.where_she_stands",
        lambda first, second, record=None: lean,
    )
    monkeypatch.setattr(skill, "_asked_of_her", _asked)
    monkeypatch.setattr(skill, "_assembled_mind", _mind)
    placed = _run(skill, _row())
    assert placed is not None, "her record answered; only the words were missing"
    assert "makes lists" in placed["why"]


def test_options_that_are_not_a_run_between_two_things_are_left_alone(monkeypatch):
    lean = Lean(toward=-0.9, first=0.4, second=0.1, because=("truth (value)",), measured=True)
    skill, _handed = _skill(monkeypatch, lean)
    labelled = [
        {"group": "q", "role": "radio", "name": name, "selector": f"#q{n}",
         "asks": "how much? [] [] []"}
        for n, name in enumerate(("agree", "neutral", "disagree"), start=1)
    ]
    assert _run(skill, labelled) is None


def test_measuring_comes_before_asking_in_a_shape():
    import inspect

    body = inspect.getsource(u._UnderstandsThePage._answer_each_question)
    assert body.index("_where_she_puts_herself") < body.index("_decide_next_actions")


def test_the_measure_is_not_tuned_to_any_instrument():
    import inspect

    from core.self import where_i_stand

    source = inspect.getsource(where_i_stand)
    for tuned in ("jungian", "extravert", "oejts", "likert", "myers"):
        assert tuned not in source.lower()
