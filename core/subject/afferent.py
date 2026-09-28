"""A low-pass reading of her organs, for the instrument. Nothing of hers reads it.

A hundred and fifty-seven of the three hundred and seventy-five columns of K_t
are read straight off a live organ: the workspace's own count of candidates, the
substrate's own weights, the self model's own ledger. This surface keeps a
leaky trace of each and hands the state reading the trace instead of the
organ's number.

It was written as her afferent channel, with her subsystems reading it through
`of`. None do: outside core/subject nothing calls it, so switched on it changes
the recording and leaves her untouched. That is rescoring on a filtered
recording, which the 28 September analysis of her time constants ruled out
unless the filter is hers (P0.17), so it lives with the instrument and no arm
of a campaign switches it on. Made hers, it would be a channel her own readers
of an organ go through; that is not built.

At the cheapest cut of `whole-s7-27dc1dda9`, carrying the columns after the
fact: nothing +0.015 with a lower bound of -0.057, the 218 alone +0.136 with
+0.042, the 157 alone +0.027 with -0.034, every column +0.240 with +0.074. Those
are readings of a filtered recording, not of her.

Off unless `AURA_AFFERENT_TURNS` asks for it.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from core.runtime.temporal_depth import Membrane

logger = logging.getLogger("Aura.Afferent")

__all__ = ["Afferent", "afferent_turns", "organ_sourced", "sensed"]


def organ_sourced() -> dict[str, frozenset[str]]:
    """Per domain, the features the schema declares as read off an organ.

    A reader returns its whole domain, and some of those columns are plain
    fields on `AuraState` that `core/runtime/state_membrane.py` already carries.
    Filtering a column at both surfaces would give it two time constants and
    make the two arms of the campaign impossible to tell apart, so the surface
    touches only what the schema says comes from an organ.
    """
    from core.subject.state import _ORGAN_READERS, _SCHEMAS

    out: dict[str, frozenset[str]] = {}
    for domain in _ORGAN_READERS:
        schema = _SCHEMAS[domain]
        out[domain] = frozenset(
            feature
            for feature, source in zip(schema.features, schema.sources, strict=True)
            if str(source).startswith("organ:")
        )
    return out


def afferent_turns() -> float:
    """How many turns of history a sensed reading holds. Zero is off."""
    raw = os.environ.get("AURA_AFFERENT_TURNS", "")
    if not raw.strip():
        return 0.0
    try:
        turns = float(raw)
    except (TypeError, ValueError):
        logger.warning("AURA_AFFERENT_TURNS is not a number (%r); nothing is sensed", raw)
        return 0.0
    return max(0.0, turns)


class Afferent:
    """The sensed copy of every organ reading, held by the runtime.

    One membrane over every column of every organ-read domain, keyed by the
    column's own name, so a domain that grows a column grows a channel here
    without anything being declared twice.
    """

    __slots__ = ("_membrane", "_mine", "frames_per_turn", "turns")

    def __init__(
        self,
        frames_per_turn: float,
        turns: float | None = None,
        mine: dict[str, frozenset[str]] | None = None,
    ) -> None:
        held = afferent_turns() if turns is None else float(turns)
        self.turns = held
        self.frames_per_turn = float(frames_per_turn)
        self._membrane = Membrane(held * frames_per_turn)
        self._mine = organ_sourced() if mine is None else mine

    @property
    def on(self) -> bool:
        return self.turns > 0.0 and self.frames_per_turn > 0.0

    def sense(self, domain: str, values: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
        """One domain's reading as she senses it. Returns the array to record.

        The organ's own array is returned untouched when the surface is off, so
        the control arm is the reading she has now.
        """
        if not self.on:
            return values
        if len(names) != values.size:
            # not a failure: a reader and its schema disagreeing is caught by
            # the width check in `read_core_state`, with a better message.
            return values
        mine = self._mine.get(domain)
        if not mine:
            return values
        out = np.array(values, dtype=np.float64, copy=True)
        for index, name in enumerate(names):
            if name not in mine:
                continue
            trace = self._membrane.feel(f"{domain}.{name}", out[index])
            if trace is not None:
                out[index] = trace
        return out

    def of(self, column: str) -> float | None:
        """What she senses of one organ column, for anything of hers that reads it."""
        return self._membrane.trace(column)

    def forget(self) -> None:
        """A fork does not inherit a body it did not live in."""
        self._membrane.forget()

    def reading(self) -> dict[str, object]:
        return {
            "schema": "aura.afferent.v1",
            "on": self.on,
            "turns": self.turns,
            "frames_per_turn": self.frames_per_turn,
            "columns": sum(len(names) for names in self._mine.values()),
            "membrane": self._membrane.reading(),
        }


def sensed(state: object, domain: str, values: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    """A domain's organ reading as she senses it, if this state has a surface."""
    surface = getattr(state, "afferent", None)
    if not isinstance(surface, Afferent):
        return values
    try:
        return surface.sense(domain, values, names)
    except (AttributeError, TypeError, ValueError) as exc:
        logger.warning("The organ reading for %s was recorded unsensed: %s", domain, exc)
        return values
