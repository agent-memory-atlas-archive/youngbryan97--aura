"""A membrane holds the trace an event leaves, so the next stage reads a history.

`core/runtime/temporal_depth.py`. The reading behind it is in
docs/WHY_IRREDUCIBILITY_FAILS.md: her channels are staircases, 77 to 92 per cent
of frames have no movement, and the cheapest cut costs 0.015 of held-out
prediction as recorded and 0.240 carried as a leaky integral at one turn.
"""

from __future__ import annotations

import math

import pytest

from core.runtime.temporal_depth import Membrane, keep_for


def test_a_time_constant_of_zero_is_the_staircase_she_has_now() -> None:
    membrane = Membrane(0.0)
    assert keep_for(0.0) == 0.0
    assert membrane.feel("a", 1.0) == 1.0
    assert membrane.feel("a", 0.0) == 0.0
    assert membrane.feel("a", 5.0) == 5.0


def test_the_first_reading_becomes_the_trace_outright() -> None:
    """Starting from zero would ramp every channel up from nothing."""
    membrane = Membrane(33.0)
    assert membrane.feel("a", 4.0) == pytest.approx(4.0)


def test_a_step_decays_towards_the_new_value_at_its_time_constant() -> None:
    membrane = Membrane(33.0)
    membrane.feel("a", 0.0)
    for _ in range(33):
        membrane.feel("a", 1.0)
    # One time constant of a step reaches 1 - 1/e.
    assert membrane.trace("a") == pytest.approx(1.0 - math.exp(-1.0), abs=0.02)


def test_a_reading_that_is_not_a_number_leaves_the_trace_where_it_was() -> None:
    """A channel that could not be read has not changed, and a zero for it is a lie."""
    membrane = Membrane(8.0)
    membrane.feel("a", 2.0)
    for value in (None, "", float("nan"), float("inf"), object()):
        assert membrane.feel("a", value) == pytest.approx(2.0)
    assert membrane.trace("a") == pytest.approx(2.0)


def test_an_unknown_channel_has_no_trace_rather_than_a_zero() -> None:
    membrane = Membrane(8.0)
    assert membrane.trace("never seen") is None
    assert membrane.trace("never seen", 0.5) == 0.5


def test_a_frame_of_channels_returns_every_trace_that_could_be_read() -> None:
    membrane = Membrane(4.0)
    out = membrane.step({"a": 1.0, "b": 2.0, "c": None})
    assert set(out) == {"a", "b"}
    assert len(membrane) == 2


def test_forgetting_leaves_a_fork_without_a_life_it_did_not_live() -> None:
    membrane = Membrane(4.0)
    membrane.step({"a": 1.0, "b": 2.0})
    membrane.forget(["a"])
    assert membrane.trace("a") is None
    assert membrane.trace("b") == pytest.approx(2.0)
    membrane.forget()
    assert len(membrane) == 0


def test_a_staircase_carried_at_one_turn_becomes_a_ramp() -> None:
    """What the measurement did to her recording, on one channel.

    A phase writes once a turn and the value sits for the other 32 frames. The
    trace of that is a ramp, and a ramp is what carries this turn into the next.
    """
    membrane = Membrane(33.0)
    staircase, ramp = [], []
    value = 0.0
    for turn in range(4):
        value = float(turn)
        for _ in range(33):
            staircase.append(value)
            ramp.append(membrane.feel("x", value))
    # The staircase takes four values; the ramp takes a new one every frame.
    assert len(set(staircase)) == 4
    assert len(set(ramp)) >= 100
    # And it lags: at the end of a turn the trace is still short of the step.
    assert ramp[-1] < staircase[-1]
    assert ramp[-1] > staircase[-34]


def test_what_it_holds_can_be_read_without_naming_a_channel() -> None:
    membrane = Membrane(33.0)
    membrane.step({"a": 1.0, "b": 2.0})
    reading = membrane.reading()
    assert reading["schema"] == "aura.membrane.v1"
    assert reading["tau_frames"] == pytest.approx(33.0)
    assert reading["channels"] == 2
    assert 0.0 < reading["keep_per_frame"] < 1.0
