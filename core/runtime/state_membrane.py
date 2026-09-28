"""Which of her channels carry a trace, and the step that gives them one.

`core/runtime/temporal_depth.py` holds the trace. This says which channels get
one and applies it, once per frame, to the state itself — so her own consumers
read the trace, and so does anything that records her. A membrane that only the
recorder saw would be a change to the measurement rather than to her.

Her channels are found in her own state: every float reachable through her
state's own fields and string-keyed maps. A float is a graded quantity; an int
is a count or an index and a trace of one is not a count. And a channel carries
a trace only once it has moved both ways, because a value that only grows is a
clock or a running total, and a trace of a clock is a late clock. Nothing here
reads the battery. The set was first the battery's own column list, which made
the organism smooth exactly what the instrument reads; tests/
test_the_organism_cannot_see_the_instrument.py failed on it, rightly.

Her kernel and her chat pipeline settle it after every phase
(`after_phase`), and the subject-core driver once a frame, so the membrane is
hers wherever she runs rather than a mechanism only the harness has.

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
    "MembraneScope",
    "after_phase",
    "her_channels",
    "membrane_turns",
    "settle",
]

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


#: How deep the walk goes into her state. Her deepest float sits five levels
#: down (`affect.markers.tangled.lifts.joy`); past that is a structure nothing
#: reads as a quantity.
_DEPTH = 6


def her_channels(state: Any) -> dict[str, float]:
    """Every float in her state, by dotted path, found by walking her own fields.

    Dataclass fields and string-keyed maps are walked; lists, sets and objects
    that are neither are not, because a position in a list is not a place a
    value can be written back to. Booleans and ints are not floats.
    """
    import dataclasses

    out: dict[str, float] = {}

    def walk(node: Any, prefix: str, depth: int) -> None:
        if depth > _DEPTH:
            return
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            items: Iterable[tuple[str, Any]] = (
                (f.name, getattr(node, f.name, None)) for f in dataclasses.fields(node)
            )
        elif isinstance(node, Mapping):
            items = ((k, v) for k, v in node.items() if isinstance(k, str) and "." not in k)
        else:
            return
        for name, value in items:
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(value, float):
                out[path] = value
            elif value is not None and not isinstance(value, (bool, int, str, bytes)):
                walk(value, path, depth + 1)

    walk(state, "", 0)
    return out


#: Which ways a channel has moved: up, down, or both.
_UP, _DOWN = 1, 2


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
        self._last: dict[str, float] = {}
        self._moved: dict[str, int] = {}

    def note(self, path: str, value: float) -> None:
        """Remember which way a channel has moved since this scope first saw it."""
        last = self._last.get(path)
        if last is not None and value != last:
            self._moved[path] = self._moved.get(path, 0) | (_UP if value > last else _DOWN)
        self._last[path] = value

    def carries(self, path: str) -> bool:
        return self._moved.get(path, 0) == _UP | _DOWN

    @property
    def channels(self) -> tuple[str, ...]:
        """The channels that carry a trace now: every float that has moved both ways."""
        return tuple(sorted(path for path, moved in self._moved.items() if moved == _UP | _DOWN))

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
            "carrying": len(self.channels),
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
    """One frame: every channel that has moved both ways becomes the trace of what it has been.

    Returns what moved, for a health report. A channel whose current value is
    not a plain number is skipped and keeps whatever its owner wrote, which is
    the same channel the recording would have read without a membrane.
    """
    if not scope.on:
        return {"on": False, "carried": 0, "skipped": 0}
    carried = skipped = refused = 0
    for path, value in her_channels(state).items():
        scope.note(path, value)
        if not scope.carries(path):
            skipped += 1
            continue
        trace = scope.membrane.feel(path, value)
        if trace is None:
            skipped += 1
            continue
        if _write(state, path, float(trace)):
            carried += 1
        else:
            refused += 1
    return {"on": True, "carried": carried, "skipped": skipped, "refused": refused}


def after_phase(state: Any, frames_per_turn: float) -> Any:
    """Settle her membrane once after a phase, when it is switched on.

    Her kernel and her chat pipeline call this after every phase they run, with
    the number of phases in their own turn as the frames a turn is worth. The
    scope rides on her state, which every phase derives from the last, so the
    traces go wherever her state goes. Returns the state it was given.
    """
    if state is None or frames_per_turn <= 0 or membrane_turns() <= 0.0:
        return state
    scope = getattr(state, "membrane", None)
    if not isinstance(scope, MembraneScope) or scope.frames_per_turn != float(frames_per_turn):
        scope = MembraneScope(float(frames_per_turn))
        try:
            state.membrane = scope
        except (AttributeError, TypeError) as exc:
            logger.warning("Her state cannot hold a membrane: %s", exc)
            return state
    try:
        settle(state, scope)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        logger.warning("Her membrane did not settle after a phase: %s", exc)
    return state
