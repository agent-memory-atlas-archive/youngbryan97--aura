"""The subject core holds what she is like, read from what she chose.

Her values score every choice she makes of what to do next, and they learn from
how the choices turned out, so they are persistent state that decides what she
does. The core read none of it. It now reads, in S, how strongly she holds each
value, how much more often than chance she took the option serving it, how
narrow her choosing has become and how often her values overrode her strongest
drive (core/agency/what_she_is_like.py).
"""

from __future__ import annotations

import pytest

from core.agency.subjective_choice import PREFERENCE_KEYS, ChoiceOption, SubjectiveChoiceEngine
from core.agency.what_she_is_like import PortraitReader
from core.state.aura_state import AuraState
from core.subject import state as core_state
from core.subject.state import Organs, read_core_state, schema


def _column(reading, name: str) -> float:
    domain, field = name.split(".", 1)
    return float(reading.values[domain][list(schema(domain).features).index(field)])


def _engine(tmp_path) -> SubjectiveChoiceEngine:
    return SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)


def _choose(engine: SubjectiveChoiceEngine, serving: str, against: str, times: int) -> None:
    for _ in range(times):
        engine.choose(
            [
                ChoiceOption(id="a", label=f"do {serving}", features={serving: 1.0}),
                ChoiceOption(id="b", label=f"do {against}", features={against: 1.0}),
            ],
            context="test",
        )


def test_the_values_are_the_engines_own():
    assert core_state._VALUES == tuple(PREFERENCE_KEYS)


def test_what_she_holds_is_in_the_core(tmp_path):
    engine = _engine(tmp_path)
    reading = read_core_state(AuraState.default(), organs=Organs(portrait=PortraitReader(engine)))
    for name, held in engine.preferences().items():
        assert _column(reading, f"S.held_{name}") == pytest.approx(held)


def test_what_she_enacts_is_in_the_core(tmp_path):
    """Offered truth against play forty times, she takes truth, which she holds higher."""
    engine = _engine(tmp_path)
    _choose(engine, "truth", "play", 40)
    reader = PortraitReader(engine)
    reading = read_core_state(AuraState.default(), organs=Organs(portrait=reader))
    assert _column(reading, "S.enacted_truth") > 0.3
    assert _column(reading, "S.enacted_play") < -0.3
    assert _column(reading, "S.enacted_beauty") == 0.0
    assert _column(reading, "S.choice_narrowness") == pytest.approx(1.0)


def test_the_portrait_is_read_again_only_when_she_has_chosen_again(tmp_path):
    engine = _engine(tmp_path)
    _choose(engine, "truth", "play", 5)
    reader = PortraitReader(engine)
    first = reader.columns()
    assert reader.columns() == first
    _choose(engine, "play", "calm", 20)
    assert reader.columns() != first


def test_what_she_is_like_feeds_one_domain():
    homes = {
        domain
        for domain in core_state.DOMAINS
        for source in schema(domain).sources
        if source.startswith("organ:portrait.")
    }
    assert homes == {"S"}


def test_the_live_organs_carry_her_portrait():
    assert Organs.live().portrait is not None
