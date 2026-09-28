"""The trace an event leaves, which is what everything downstream reads.

Her subsystems write what just happened and then the value sits until the next
turn replaces it. Measured on `whole-s7-27dc1dda9`: 77 to 92 per cent of frames
have no movement in a given domain, the steps have a kurtosis of 142 to 10,699
against a Gaussian's 3, and one column carries 94.6 per cent of A's step
variance. So each channel is a staircase, and the coupling between her domains
is in which frames step rather than by how much — a hard move in P makes a hard
move in I 3.7 times more likely on the next frame, W to S 3.4, A to N 4.1.

A membrane is what a body puts between an event and everything that reads it.
The event arrives, the trace rises, and the trace decays; what the next stage
sees is the recent history of the signal rather than its last sample. Every
synapse is a low-pass filter for the same reason, and the population rates that
integration is computed over are the filtered signal, not the spikes.

Carried as a leaky integral at one turn, her cheapest cut costs 0.240 of
held-out prediction where the staircase costs 0.015, and five nulls stay at zero
under the same filter, including her own domains dealt to other turns
(docs/WHY_IRREDUCIBILITY_FAILS.md). The coupling is there; a channel with no
time constant has no window to integrate it over.

This holds the trace. It decides nothing about which channels have one; a
caller names those, so the set is a declaration that can be read rather than a
behaviour spread through the writers.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["Membrane", "keep_for", "ONE_TURN_IS_THE_UNIT"]

#: The time constant is given in frames and the caller takes it from the run's
#: own clock. `frames_per_turn` on the seed-7 recording is 33, and one turn is
#: the interval a phase writes on, so a trace of one turn is a channel that
#: carries what this turn did into the next. Nothing here chooses a number.
ONE_TURN_IS_THE_UNIT = 1.0


def keep_for(tau_frames: float) -> float:
    """How much of the trace survives one frame, for a time constant in frames.

    A time constant of zero keeps nothing, which is the staircase she has now,
    so a caller can turn the membrane off by asking for it.
    """
    try:
        tau = float(tau_frames)
    except (TypeError, ValueError) as exc:
        logger.debug("time constant %r is not a number, so the trace keeps nothing (%s: %s)",
                     tau_frames, type(exc).__name__, exc)
        return 0.0
    if not math.isfinite(tau) or tau <= 0.0:
        return 0.0
    return float(math.exp(-1.0 / tau))


class Membrane:
    """A leaky trace per named channel, stepped once per frame.

    The first reading of a channel becomes its trace outright. Starting from
    zero would make every channel ramp up from nothing over its first time
    constant, which is an artefact of when the membrane was built rather than
    anything about her.
    """

    __slots__ = ("_keep", "_tau", "_trace")

    def __init__(self, tau_frames: float) -> None:
        self._tau = float(tau_frames)
        self._keep = keep_for(tau_frames)
        self._trace: dict[str, float] = {}

    @property
    def tau_frames(self) -> float:
        return self._tau

    @property
    def keep(self) -> float:
        return self._keep

    def feel(self, name: str, value: Any) -> float | None:
        """One channel, one frame. Returns the trace, or None if there is none.

        A value that is not a finite number leaves the trace where it is: a
        channel that could not be read this frame has not changed, and writing
        a zero for it would be a reading nothing took.
        """
        try:
            reading = float(value)
        except (TypeError, ValueError):
            return self._trace.get(name)
        if not math.isfinite(reading):
            return self._trace.get(name)
        held = self._trace.get(name)
        if held is None or self._keep <= 0.0:
            self._trace[name] = reading
        else:
            self._trace[name] = self._keep * held + (1.0 - self._keep) * reading
        return self._trace[name]

    def step(self, readings: Mapping[str, Any]) -> dict[str, float]:
        """A frame's worth of channels. Returns every trace that moved."""
        out: dict[str, float] = {}
        for name, value in readings.items():
            trace = self.feel(name, value)
            if trace is not None:
                out[name] = trace
        return out

    def trace(self, name: str, default: float | None = None) -> float | None:
        return self._trace.get(name, default)

    def forget(self, names: Iterable[str] | None = None) -> None:
        """Drop traces, so a fork does not inherit a life it did not live."""
        if names is None:
            self._trace.clear()
            return
        for name in names:
            self._trace.pop(name, None)

    def reading(self) -> dict[str, Any]:
        """What it is holding, for a health report or a recording."""
        return {
            "schema": "aura.membrane.v1",
            "tau_frames": round(self._tau, 4),
            "keep_per_frame": round(self._keep, 6),
            "channels": len(self._trace),
        }

    def __len__(self) -> int:
        return len(self._trace)
