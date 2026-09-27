"""The self-review reports what it finds and files nothing nobody reads.

It filed an "architectural_review" intent whenever phi was low or entropy high,
for a self-modification engine with no way to take one. Nothing in the tree
read that intent type, and the list keeps its last twenty, so each review
pushed out an intent something does read. Its ten-minute cadence ran on the
event loop's clock, the machine's, and in a measurement run it fired on the
first turn of every arm.
"""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

from core.kernel import self_review
from core.kernel.self_review import SelfReviewPhase
from core.state.aura_state import AuraState


def _phase(reading: dict[str, float]) -> SelfReviewPhase:
    return SelfReviewPhase(SimpleNamespace(loop_state=lambda: dict(reading)))


def _at(monkeypatch, clock: list[float]) -> None:
    monkeypatch.setattr(self_review, "time", SimpleNamespace(time=lambda: clock[0]))


def test_a_low_phi_review_leaves_her_intents_as_they_were(monkeypatch):
    clock = [1_000_000.0]
    _at(monkeypatch, clock)
    phase = _phase({"phi": 0.05, "entropy": 0.3})
    state = AuraState.default()
    state.cognition.pending_intents = [{"type": "autotelic_objective", "domain": "tides"}]
    for _ in range(3):
        state = asyncio.run(phase.execute(state))
        clock[0] += 601.0
    assert state.cognition.pending_intents == [{"type": "autotelic_objective", "domain": "tides"}]


def test_it_warns_when_the_finding_starts_and_not_on_every_review(monkeypatch, caplog):
    clock = [1_000_000.0]
    _at(monkeypatch, clock)
    phase = _phase({"phi": 0.05, "entropy": 0.3})
    state = AuraState.default()
    with caplog.at_level(logging.DEBUG, logger="Aura.SelfReview"):
        for _ in range(4):
            state = asyncio.run(phase.execute(state))
            clock[0] += 601.0
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_it_reviews_on_her_clock(monkeypatch):
    clock = [1_000_000.0]
    _at(monkeypatch, clock)
    phase = _phase({"phi": 0.5, "entropy": 0.3})
    state = AuraState.default()
    asyncio.run(phase.execute(state))
    first = phase._last_review_ts
    clock[0] += 599.0
    asyncio.run(phase.execute(state))
    assert phase._last_review_ts == first
    clock[0] += 2.0
    asyncio.run(phase.execute(state))
    assert phase._last_review_ts == clock[0]
