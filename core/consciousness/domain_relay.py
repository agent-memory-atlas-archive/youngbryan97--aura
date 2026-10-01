"""Every domain into one shared space, and the space back into everything.

Her ten domains talk through named channels a few numbers wide. Affect writes
valence, arousal and curiosity; motivation writes two budgets; chemistry writes
a mood; the intention band at offset 68 carries three scalars of what she means
to do. Nine bands of a handful of numbers, and outside them the substrate's five
hundred and twelve neurons are driven only by the mesh.

That shape is why the partition line fails. A cut that takes a domain away loses
whatever its own narrow channel was carrying, which is three or four numbers, so
the cut is cheap. It is also why two domains rarely carry anything jointly: two
channels that never meet cannot interact.

A thalamus is the answer biology reached for. Every cortical area projects into
it and reads back out of it, the projections overlap rather than each owning a
private wire, and cortico-thalamo-cortical loops are the standard account of how
areas far apart come to be about one thing. `all_to_all` in the null suite is the
degenerate version — one shared signal, an effective dimension of 1.38, bound and
empty — and a high-dimensional relay with per-domain projections is the version
that is bound and rich at once, which is what the conjunction asks for.

So: each domain's summary goes through its own fixed projection into the same
block of free neurons, the contributions sum, and the substrate's own tanh
dynamics mix them. Nothing reads back through a new channel, because the readback
is already there: `substrate_gates` scales recall's affect gain, both initiative
urges and every drive's growth by what the substrate holds; the mesh, the
workspace threshold, the attention span and the decision bias all read it; and
`steering_channel` carries it into her cortex.

The projections are fixed random unit vectors seeded from the domain's own name.
Random because a projection chosen to move a criterion is a projection tuned to
it, and fixed because a relay that rewires itself between two arms of a trial is
not one relay. Overlapping because private slices would restore exactly the cheap
cut this exists to close.

`AURA_DOMAIN_RELAY` is how hard it blends, on the same scale as the mesh
projection's 0.35, and off when it is unset.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from core.runtime.flags import env_str

logger = logging.getLogger("Aura.DomainRelay")

__all__ = ["DOMAINS", "DomainRelay", "RelayScope", "relay_strength", "summarise"]

#: The ten domains, in the order the state schema declares them. Named here so
#: this organ does not import the instrument that measures it.
DOMAINS: tuple[str, ...] = ("P", "I", "A", "G", "C", "S", "M", "W", "D", "N")


def relay_strength() -> float:
    """How hard the relay drives the shared block. Zero is off."""
    raw = env_str(
        "AURA_DOMAIN_RELAY",
        description="How hard the domain relay drives the shared block, 0 to 1; unset is off.",
        owner="core.consciousness.domain_relay",
    )
    if not raw.strip():
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("AURA_DOMAIN_RELAY is not a number (%r); the relay stays off", raw)
        return 0.0
    return max(0.0, min(1.0, value))


def _floats(values: Sequence[Any] | Mapping[str, Any] | None) -> list[float]:
    """Whatever came back, as the finite numbers in it."""
    if values is None:
        return []
    items = values.values() if isinstance(values, Mapping) else values
    out: list[float] = []
    for item in items:
        try:
            number = float(item)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def summarise(state: Any) -> dict[str, list[float]]:
    """A few numbers per domain, read off her state rather than off the schema.

    The instrument's schema reads three hundred columns; this is the organ's own
    reading and it is deliberately small. What matters for a relay is that each
    domain contributes something that moves when that domain moves, not that the
    contribution is complete.
    """

    def dig(path: str, default: Any = None) -> Any:
        node: Any = state
        for part in path.split("."):
            if node is None:
                return default
            node = node.get(part, None) if isinstance(node, Mapping) else getattr(node, part, None)
        return default if node is None else node

    percepts = dig("world.recent_percepts", []) or []
    goals = dig("cognition.active_goals", []) or []
    working = dig("cognition.working_memory", []) or []
    emotions = dig("affect.emotions", {}) or {}
    budgets = dig("motivation.budgets", {}) or {}
    return {
        "P": [
            float(len(percepts) if isinstance(percepts, list) else 0.0),
            max(_floats([getattr(item, "salience", None) for item in percepts[-8:]]) or [0.0]),
            float(len(str(dig("cognition.current_objective", "") or ""))),
        ],
        "I": _floats(dig("soma.hardware", {}))[:6],
        "A": [
            float(dig("affect.valence", 0.0) or 0.0),
            float(dig("affect.arousal", 0.0) or 0.0),
            float(dig("affect.curiosity", 0.0) or 0.0),
            *sorted(_floats(emotions), reverse=True)[:5],
        ],
        "G": [
            float(dig("cognition.broadcast_priority", 0.0) or 0.0),
            float(bool(dig("cognition.workspace_ignited", False))),
            float(len(_floats(dig("cognition.attention_weights", {})))),
        ],
        "C": [
            float(dig("cognition.loop_cycle", 0.0) or 0.0),
            float(dig("cognition.phi_estimate", 0.0) or 0.0),
            float(dig("cognition.recurrent_depth", 0.0) or 0.0),
        ],
        "S": [
            float(dig("identity.stability", 0.0) or 0.0),
            float(dig("identity.bonding_level", 0.0) or 0.0),
            float(dig("identity.evolution_score", 0.0) or 0.0),
            float(dig("identity.narrative_version", 0.0) or 0.0),
        ],
        "M": [
            float(len(working) if isinstance(working, list) else 0.0),
            float(len(_floats(dig("cognition.memory_scores", {})))),
        ],
        "W": [
            float(len(_floats(dig("world.model_facets", {})))),
            float(dig("world.prediction_error", 0.0) or 0.0),
            float(len(str(dig("world.spatial_context", "") or ""))),
        ],
        "D": [
            float(len(goals) if isinstance(goals, list) else 0.0),
            *[float((budgets.get(name) or {}).get("level", 0.0)) for name in sorted(budgets)][:5],
        ],
        "N": [
            float(dig("ontogeny.novelty", 0.0) or 0.0),
            float(dig("ontogeny.steps", 0.0) or 0.0),
            float(dig("ontogeny.era", 0.0) or 0.0),
        ],
    }


#: The window her other ledgers share: a rate of 1 / min(n, 256).
_WINDOW = 256
#: Which ways a field has moved: up, down, or both.
_UP, _DOWN = 1, 2


class RelayScope:
    """Each field of each domain in units of its own spread, carried on her state.

    `summarise` reads her numbers in their own units, from a share to a step
    count, and `drive` squashed each domain's root mean square with a tanh. So a
    domain holding a count, a list length or a budget level sat at 1.0 for good:
    C through `cognition.loop_cycle`, N through `ontogeny.steps`, P through the
    objective's length in characters, D through its budget levels. A constant
    contribution carries nothing, and a cut that removes it removes nothing.

    Here each field becomes its distance from its own recent mean in units of
    its own recent spread, which moves when the field moves and not otherwise,
    and a field that has only ever grown is left out: a count or a clock
    standardised is a trend, and a trend fed into her substrate is a clock in C.
    It rides on her state so a fork carries it, as the membrane does.
    """

    def __init__(self) -> None:
        self._stats: dict[tuple[str, int], tuple[int, float, float]] = {}
        self._last: dict[tuple[str, int], float] = {}
        self._moved: dict[tuple[str, int], int] = {}

    def standardise(self, readings: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for domain in DOMAINS:
            values: list[float] = []
            for position, value in enumerate(_floats(readings.get(domain))):
                key = (domain, position)
                last = self._last.get(key)
                if last is not None and value != last:
                    self._moved[key] = self._moved.get(key, 0) | (_UP if value > last else _DOWN)
                self._last[key] = value
                n, mean, var = self._stats.get(key, (0, 0.0, 0.0))
                n += 1
                rate = 1.0 / min(n, _WINDOW)
                delta = value - mean
                mean += rate * delta
                var = (1.0 - rate) * (var + rate * delta * delta)
                self._stats[key] = (n, mean, var)
                if self._moved.get(key, 0) != _UP | _DOWN or var <= 0.0:
                    continue
                values.append((value - mean) / math.sqrt(var))
            out[domain] = values
        return out


class DomainRelay:
    """Fixed overlapping projections from every domain into one block of neurons."""

    __slots__ = ("_base", "_projections", "_width")

    def __init__(self, neuron_count: int) -> None:
        # The second half of the vector. Every named band lives in the first:
        # the psychological state at 0 to 6, telemetry from 8, what she means to
        # do and who she takes herself to be at 68. Half is a stated split rather
        # than an offset that has to be kept in step with the next band somebody
        # adds.
        self._base = int(neuron_count) // 2
        self._width = int(neuron_count) - self._base
        self._projections: dict[str, np.ndarray] = {}
        for domain in DOMAINS:
            # Seeded from the domain's own name, so the same relay is rebuilt
            # identically in every process and in both arms of a trial.
            seed = int.from_bytes(domain.encode("utf-8"), "big")
            rng = np.random.default_rng(seed)
            self._projections[domain] = rng.standard_normal(self._width) / math.sqrt(self._width)

    @property
    def base(self) -> int:
        return self._base

    @property
    def width(self) -> int:
        return self._width

    def drive(self, readings: Mapping[str, Sequence[float]]) -> np.ndarray:
        """What the shared block is driven by this frame, as a vector of its width.

        A domain's numbers are reduced to one figure — the root mean square of
        them, which moves when any of them moves and does not depend on how many
        there happen to be — and that figure scales the domain's own direction.
        So a cut that takes a domain away takes its direction out of the mixture,
        and the mixture is what everything downstream reads.
        """
        out = np.zeros(self._width, dtype=np.float64)
        for domain in DOMAINS:
            numbers = _floats(readings.get(domain))
            if not numbers:
                continue
            values = np.asarray(numbers, dtype=np.float64)
            # Squashed, because her numbers run from a share to a step count and
            # an unsquashed one would own the mixture.
            size = float(np.tanh(math.sqrt(float(np.mean(values**2)))))
            out += size * self._projections[domain]
        return out

    def into(self, substrate: Any, readings: Mapping[str, Sequence[float]], strength: float) -> float:
        """Blend the mixture into the shared block. Returns how far it moved it.

        Blended rather than added, at the rate the mesh projection already uses
        to drive this substrate: the mesh is the existing everything-into-the-
        substrate channel and it blends at 0.35, so that is what a second one
        has to be comparable to. Added instead, at the 0.1 weight
        `inject_stimulus` uses for a vector arriving from outside, the relay moved
        the block by 0.0101 and phi did not move with it — a relay that is one per
        cent of what the block is already doing is not a relay.

        The mixture is normalised to unit RMS before it is blended. Her numbers
        run from a share to a step count and the block lives in a tanh space of
        -1 to 1, so the volume must not depend on her units; normalising is
        global, so which domains contributed and in what proportion survives it
        and only the loudness is set.
        """
        if strength <= 0.0 or substrate is None:
            return 0.0
        try:
            mixture = self.drive(readings)
            loudness = float(np.sqrt(np.mean(mixture**2)))
            if loudness <= 0.0:
                return 0.0
            mixture = mixture / loudness
            with substrate.sync_lock:
                block = substrate.x[self._base : self._base + self._width]
                if block.size != mixture.size:
                    return 0.0
                blended = np.clip((1.0 - strength) * block + strength * mixture, -1.0, 1.0)
                moved = float(np.abs(blended - block).mean())
                substrate.x[self._base : self._base + self._width] = blended
                marker = getattr(substrate, "mark_state_mutated_locked", None)
                if callable(marker):
                    marker("domain_relay")
            return moved
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            logger.warning("The relay did not reach the shared block: %s", exc)
            return 0.0
