"""A mood nothing has appraised is a configured baseline, and is shown as one.

L08, 2026-09-29: the mind page showed "How she feels" from `/api/inner-state`,
which passed the affect engine's state through without saying whether anything
that happened had moved it. Before the first appraisal the markers hold their
configured baseline and its drift, so the page could show a prior as a
feeling; and where the state had no valence at all, the endpoint reported 0,
which the page drew as a steady mood.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core.affect.damasio_v2 import AffectEngineV2

ROOT = Path(__file__).resolve().parents[1]


def _observed(event_id: str) -> dict:
    return {
        "event_id": event_id,
        "source": "mood_measured_test",
        "intensity": 1.0,
        "evidence": {"kind": "test_observation"},
        "appraisal": {"v": 0.7, "a": 0.6, "e": 0.8},
    }


def test_a_fresh_engine_says_nothing_has_been_appraised() -> None:
    state = AffectEngineV2().get_state_sync()
    assert state["appraised"] is False
    assert state["stimuli_appraised"] == 0
    assert state["last_appraised_at"] is None


@pytest.mark.asyncio
async def test_an_applied_stimulus_is_counted_and_a_duplicate_is_not() -> None:
    engine = AffectEngineV2()
    await engine.react("novel_stimulus", _observed("mood-1"))
    await engine.react("novel_stimulus", _observed("mood-1"))
    state = engine.get_state_sync()
    assert state["appraised"] is True
    assert state["stimuli_appraised"] == 1
    assert state["last_appraised_at"] is not None


def test_the_endpoint_does_not_invent_a_valence() -> None:
    source = (ROOT / "interface" / "routes" / "inner_state.py").read_text(encoding="utf-8")
    assert 'getattr(state, "valence", 0)' not in source
    assert 'getattr(state, "valence", None)' in source


def test_the_page_shows_an_unappraised_mood_as_not_yet_measured() -> None:
    page = (ROOT / "interface" / "static" / "mind.html").read_text(encoding="utf-8")
    assert "d.affect.appraised===false" in page
    # A null valence is worded rather than drawn: "settling in".
    assert 'if(v==null) return "settling in"' in page
