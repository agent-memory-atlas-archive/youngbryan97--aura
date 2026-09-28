"""Which of her channels carry a trace, and the step that gives them one.

`core/runtime/temporal_depth.py` holds the trace. This says which channels get
one and applies it, once per frame, to the state itself — so her own consumers
read the trace, and so does anything that records her. A membrane that only the
recorder saw would be a change to the measurement rather than to her.

Only channels that are a plain number at a plain dotted path on `AuraState` are
eligible. A reading that has to be dug out of a list, or asked of an organ, is
not a field anything can write back to, and a membrane that could not write back
would be a reading nothing takes.

One time constant, one turn, for every channel. A turn is the interval her
phases write on and it is the run's own `frames_per_turn`, so nothing here is
chosen.

A second constant was tried first, eight turns for the domains the schema
declares as carrying across turns, on the argument that a campaign's condition
cycle is eight. It was the one number here that was picked rather than measured,
and it cost: every column carried at one turn reads phi +0.240 at the cheapest
cut, and the same columns at one turn for the fast domains and eight for the slow
read +0.013 with a lower bound of -0.401, worse than carrying nothing. A slow
channel held for eight turns is a constant, and a constant can be cut away for
free. So there is one constant and it is hers.

Off unless a caller asks for it. `AURA_MEMBRANE_TURNS=0` is the staircase she has
now, which is the control arm of the campaign that reads this.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from typing import Any

from core.runtime.temporal_depth import Membrane

logger = logging.getLogger("Aura.StateMembrane")

__all__ = [
    "FRAMES_PER_TURN_DEFAULT",
    "MembraneScope",
    "eligible_channels",
    "membrane_turns",
    "settle",
]

#: Frames a turn holds when nothing has measured one yet. The clock of every
#: seed-7 recording reads 33, and a null architecture has no turns of its own to
#: read, so this is what a toy is carried at when she is carried at one turn.
FRAMES_PER_TURN_DEFAULT: float = 33.0


def membrane_turns() -> float:
    """How many turns a fast channel holds, from the environment. Zero is off."""
    raw = os.environ.get("AURA_MEMBRANE_TURNS", "")
    if not raw.strip():
        return 0.0
    try:
        turns = float(raw)
    except (TypeError, ValueError):
        logger.warning("AURA_MEMBRANE_TURNS is not a number (%r); the membrane stays off", raw)
        return 0.0
    return max(0.0, turns)


#: Built once. A scope is carried on her state and a fork deep-copies it, so a
#: per-instance table of two hundred columns would be copied on every arm of
#: every trial for no reason; the table is the same for all of them.
_CHANNELS: dict[str, str] | None = None


def eligible_channels() -> dict[str, str]:
    """Every schema column whose value is a plain number at a writable path.

    Returns the column name against the dotted path it reads, so the set is a
    table that can be printed rather than a behaviour spread through writers.
    """
    global _CHANNELS

    if _CHANNELS is not None:
        return _CHANNELS
    from core.subject.state import _SCHEMAS, DOMAINS

    out: dict[str, str] = {}
    for domain in DOMAINS:
        schema = _SCHEMAS[domain]
        for feature, source in zip(schema.features, schema.sources, strict=True):
            path = str(source)
            if path.startswith("organ:") or "[" in path or "*" in path:
                continue
            out[f"{domain}.{feature}"] = path
    _CHANNELS = out
    return out


class MembraneScope:
    """The membrane, carried on her state so a fork carries it too.

    It used to live on the runtime, which `restore` does not touch, so the
    second arm of a paired trial began with the first arm's traces: a sham arm
    that was not the same arm. Her state is deep-copied from the snapshot, so
    on the state both arms begin from the anchor's own history.
    """

    def __init__(self, frames_per_turn: float, turns: float | None = None) -> None:
        held = membrane_turns() if turns is None else float(turns)
        self.frames_per_turn = float(frames_per_turn)
        self.turns = held
        self.membrane = Membrane(held * frames_per_turn)

    @property
    def channels(self) -> dict[str, str]:
        return eligible_channels()

    @property
    def on(self) -> bool:
        return self.turns > 0.0

    def forget(self, columns: Iterable[str] | None = None) -> None:
        self.membrane.forget(columns)

    def reading(self) -> dict[str, Any]:
        return {
            "schema": "aura.state_membrane.v1",
            "on": self.on,
            "turns": self.turns,
            "frames_per_turn": self.frames_per_turn,
            "eligible": len(self.channels),
            "membrane": self.membrane.reading(),
        }


def _read(root: Any, path: str) -> Any:
    node = root
    for part in path.split("."):
        if node is None:
            return None
        node = node.get(part, None) if isinstance(node, Mapping) else getattr(node, part, None)
    return node


def _write(root: Any, path: str, value: float) -> bool:
    """Put the trace back where the reading came from. False if nothing took it."""
    parts = path.split(".")
    node = root
    for part in parts[:-1]:
        if node is None:
            return False
        node = node.get(part, None) if isinstance(node, Mapping) else getattr(node, part, None)
    if node is None:
        return False
    last = parts[-1]
    try:
        if isinstance(node, Mapping):
            node[last] = value
        else:
            setattr(node, last, value)
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        # not a failure: a frozen dataclass or a property with no setter is a
        # field she does not hold, and it keeps the value its owner computed.
        logger.debug("A trace had nowhere to go at %s: %s", path, exc)
        return False
    return True


def settle(state: Any, scope: MembraneScope) -> dict[str, Any]:
    """One frame: every eligible channel becomes the trace of what it has been.

    Returns what moved, for a health report. A channel whose current value is
    not a plain number is skipped and keeps whatever its owner wrote, which is
    the same channel the recording would have read without a membrane.
    """
    if not scope.on:
        return {"on": False, "carried": 0, "skipped": 0}
    carried = skipped = refused = 0
    for column, path in scope.channels.items():
        value = _read(state, path)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            skipped += 1
            continue
        trace = scope.membrane.feel(column, value)
        if trace is None:
            skipped += 1
            continue
        if _write(state, path, float(trace)):
            carried += 1
        else:
            refused += 1
    return {"on": True, "carried": carried, "skipped": skipped, "refused": refused}
