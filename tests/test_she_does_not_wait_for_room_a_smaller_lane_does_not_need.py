"""She polled a 202 for sixty-eight seconds with a lane that fitted sitting idle.

Live 2026-09-27: the cortex was refused its spawn at 11.9GB of headroom against
24GB required, six attempts, RAM at 81%, `Conversation: FAIL`. The admission
path treats a deferral as "not yet" and waits it out, which is right for the
case it was written for — a live deferral short by under a gigabyte. Nothing was
going to free twelve gigabytes inside that wait, and the tertiary tier's own
floor (under 92% pressure, at least 6GB available) was satisfied the whole time.

So the wait asks the gate, in the gate's own numbers, whether the smaller lane
is admissible already. If it is, there is nothing to wait for.
"""

from __future__ import annotations

import pytest

from interface.routes import chat_foreground_lane as lane_module


def _snapshot(**kw):
    base = {
        "tier": "tertiary",
        "measured": True,
        "can_admit": True,
        "available_gb": 11.9,
        "min_available_gb": 6.0,
        "pressure_pct": 81.4,
        "max_pressure_pct": 92.0,
    }
    base.update(kw)
    return base


@pytest.fixture
def gate(monkeypatch):
    def _set(snapshot):
        from core.brain import inference_gate

        monkeypatch.setattr(
            inference_gate.InferenceGate,
            "_headroom_snapshot",
            staticmethod(lambda tier="primary": dict(snapshot)),
        )

    return _set


def test_the_incident_numbers_mean_the_smaller_lane_fits(gate):
    """11.9GB available against a 6GB floor, 81.4% against a 92% ceiling."""
    gate(_snapshot())
    assert lane_module._a_smaller_lane_already_fits() is True


def test_a_tier_the_gate_refuses_is_still_worth_waiting_for(gate):
    gate(_snapshot(can_admit=False))
    assert lane_module._a_smaller_lane_already_fits() is False


def test_an_unmeasured_gate_keeps_the_old_behaviour(gate):
    """Waiting is what it did before; an absent measurement does not change it."""
    gate(_snapshot(measured=False))
    assert lane_module._a_smaller_lane_already_fits() is False


def test_a_gate_that_raises_keeps_the_old_behaviour(monkeypatch):
    from core.brain import inference_gate

    def _raise(tier="primary"):
        raise RuntimeError("no monitor")

    monkeypatch.setattr(
        inference_gate.InferenceGate, "_headroom_snapshot", staticmethod(_raise)
    )
    assert lane_module._a_smaller_lane_already_fits() is False


def test_it_asks_about_the_tier_that_would_answer(monkeypatch):
    asked: list[str] = []

    from core.brain import inference_gate

    def _record(tier="primary"):
        asked.append(tier)
        return _snapshot()

    monkeypatch.setattr(
        inference_gate.InferenceGate, "_headroom_snapshot", staticmethod(_record)
    )
    lane_module._a_smaller_lane_already_fits()
    assert asked == ["tertiary"]


def test_the_wait_is_conditioned_on_it():
    """The deferral branch must consult it rather than waiting unconditionally."""
    import inspect

    source = inspect.getsource(lane_module._admit_to_foreground_lane)
    assert "_a_smaller_lane_already_fits()" in source
    assert "not _a_smaller_lane_already_fits()" in source


def test_nothing_here_chooses_a_threshold():
    """Both numbers are the gate's own."""
    import inspect

    source = inspect.getsource(lane_module._a_smaller_lane_already_fits)
    assert "min_available_gb" in source and "max_pressure_pct" in source
    assert "can_admit" in source
