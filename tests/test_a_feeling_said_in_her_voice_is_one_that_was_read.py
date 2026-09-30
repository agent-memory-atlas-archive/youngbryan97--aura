"""A feeling said in the first person is one something measured.

L08, 29 Sep: the phenomenal-now claim and its interior narrative are the
first-person sentences written into `cognition.phenomenal_state` each tick.
With no substrate and no affect module to read, both were still built from
the defaults on `SubstrateSummary` — "I am aware of the present moment, still
in this moment", "Very still. Almost empty" — so a configured value was said as
something she felt. The engine already knew: `tick` computed whether the
substrate had been read and then did not use it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.consciousness.phenomenal_now import (
    AttentionSummary,
    PhenomenalNowEngine,
    QualityMetrics,
    SubstrateSummary,
    TemporalBinding,
    WorkspaceSummary,
)
from core.container import ServiceContainer

FEELING_WORDS = ("feeling", "still", "warm", "steady", "engaged", "content", "grey", "quiet")


def _say(summary: SubstrateSummary) -> tuple[str, str]:
    engine = PhenomenalNowEngine()
    attention = AttentionSummary()
    workspace = WorkspaceSummary()
    claim = engine._generate_phenomenal_claim(attention, summary, workspace)
    narrative = engine._generate_interior_narrative(
        attention, summary, workspace, TemporalBinding(), QualityMetrics()
    )
    return claim, narrative


def test_nothing_read_is_not_said_as_a_feeling() -> None:
    engine = PhenomenalNowEngine()
    summary, _ = engine._pull_substrate()
    assert summary.felt_measured is False
    assert summary.emotion_measured is False
    claim, narrative = _say(summary)
    assert not any(word in claim.lower() for word in FEELING_WORDS), claim
    assert "Very still" not in narrative and "No particular color" not in narrative, narrative


def test_a_read_substrate_is_still_said() -> None:
    substrate = SimpleNamespace(
        get_substrate_affect=lambda: {"valence": 0.6, "arousal": 0.5, "energy": 0.8, "volatility": 0.1}
    )
    ServiceContainer.register_instance("conscious_substrate", substrate, required=False)
    summary, _ = PhenomenalNowEngine()._pull_substrate()
    assert summary.felt_measured is True
    claim, narrative = _say(summary)
    assert summary.felt_quality in claim
    assert narrative.split(".")[0], narrative


def test_an_emotion_is_said_only_when_the_affect_module_named_it() -> None:
    affect = SimpleNamespace(dominant_emotion="curious")
    ServiceContainer.register_instance("affect_module", affect, required=False)
    summary, _ = PhenomenalNowEngine()._pull_substrate()
    assert summary.emotion_measured is True and summary.felt_measured is False
    claim, narrative = _say(summary)
    assert "feeling curious" in claim
    assert "There's a pull toward understanding." in narrative


@pytest.mark.parametrize("emotion", ["joy", "fear"])
def test_a_default_emotion_is_not_voiced_over_a_read_substrate(emotion: str) -> None:
    summary = SubstrateSummary(dominant_emotion=emotion, emotion_measured=False)
    claim, narrative = _say(summary)
    assert emotion not in claim
    assert "Brightness in the field" not in narrative and "Something tensed" not in narrative
