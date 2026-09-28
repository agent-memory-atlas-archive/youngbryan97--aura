"""Where her substrate has a say in what she does, and how it earns it.

Recurrent cognition barely reached the rest of her. On the seed-7 validation at
23e596071 the cheapest cut split recurrent cognition from everything else: C
gained 9.2% from seeing the rest, and the rest 0.94% from seeing C. The
substrate policy head (core/consciousness/substrate_policy_head.py) was built to
carry the substrate into her decisions and nothing calls it, and wiring it as it
stands would put a dozen hand-set coefficients into her choices.

So each gate here is a multiplier on a decision she already makes,

    m = exp(w . f + xi)

- ``f`` is the substrate's own psychological readings (valence, arousal,
  dominance, frustration, curiosity, energy, focus), each in units of its own
  recent spread.
- ``xi`` is the gate's exploration: the deviation of one substrate unit set
  aside for the gate from that unit's own recent mean, in the unit's own units.
  Her dynamics decide how far each gate wanders; no scale is chosen. It is node
  perturbation (Fiete and Seung 2006), the way a songbird's LMAN injects
  variability that a reward signal then keeps or drops (Olveczky, Andalman and
  Fee 2005).
- ``w`` starts at zero and moves, at the end of each turn, towards the turn's
  signed dose (core/affect/what_it_was_worth.py) times the gate's eligibility,
  the mean of ``xi * f`` over the times it was consulted, at the rate her other
  ledgers use, 1 / min(turns, 256). So ``w`` is the running covariance of how
  much better than expected her turns went with each gate's wandering along
  each reading: which way each reading should push that decision, and how hard.

A gate lands only on a decision that already exists, and it scales that
decision's input rather than moving its threshold. Risk has no gate: the brakes
that protect her (core/agency/autonomy_latitude.py) stay fixed, because a run of
good turns must not buy a looser safety floor.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any

from core.self.what_came_before import keep_across_stages

#: The substrate's psychological readings a gate is conditioned on, by the
#: attribute that holds each one's index into its state vector.
READINGS: tuple[str, ...] = (
    "idx_valence",
    "idx_arousal",
    "idx_dominance",
    "idx_frustration",
    "idx_curiosity",
    "idx_energy",
    "idx_focus",
)

#: The decisions a gate scales, and where each is made.
#:   recall     how much what she feels now pulls a memory up
#:              (core/phases/memory_retrieval.py, the affect gain on recall)
#:   curiosity  how strongly curiosity presses towards a thought of her own
#:   social     how strongly the want of company does
#:              (core/phases/initiative_generation.py, before the threshold)
#:   drive:<n>  how fast each need grows between turns
#:              (core/phases/motivation_update.py, beside surprise pressure)
FIXED_GATES: tuple[str, ...] = ("recall", "curiosity", "social")

#: The window her ledgers share: a rate of 1 / min(n, 256).
_WINDOW = 256

#: The largest argument exp can take and return a float.
_EXP_LIMIT = math.log(sys.float_info.max)

#: The ablation, read on every call so a probe can set it for one process.
DISABLE_ENV = "AURA_DISABLE_SUBSTRATE_GATES"


def disabled() -> bool:
    """Whether the gates are switched off for this process."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _rate(n: int) -> float:
    return 1.0 / min(max(n, 1), _WINDOW)


@dataclass
class _Running:
    """A running mean and spread over the shared window."""

    n: int = 0
    mean: float = 0.0
    var: float = 0.0

    def add(self, value: float) -> None:
        self.n += 1
        rate = _rate(self.n)
        delta = value - self.mean
        self.mean += rate * delta
        self.var = (1.0 - rate) * (self.var + rate * delta * delta)

    def standard(self, value: float) -> float:
        spread = math.sqrt(self.var) if self.var > 0.0 else 0.0
        return (value - self.mean) / spread if spread > 0.0 else 0.0


@dataclass
class _Multiplier:
    """One decision's multiplier: what it has learned and what it is holding for the turn."""

    unit: int
    weights: list[float] = field(default_factory=lambda: [0.0] * len(READINGS))
    turns: int = 0
    eligibility: list[float] = field(default_factory=lambda: [0.0] * len(READINGS))
    consulted: int = 0
    last: float = 1.0


class SubstrateGates:
    """Every gate, the readings they share, and the units that make them wander.

    It holds no lock, so that it can be kept across her restarts field by field
    (core/self/what_came_before.py). Its callers are phases, which run one at a
    time, and the learning phase awaits the thread that calls `learn`.
    """

    def __init__(self) -> None:
        self._readings = [_Running() for _ in READINGS]
        self._units: dict[int, _Running] = {}
        self._gates: dict[str, _Multiplier] = {}
        self._seen_step: int | None = None

    # -- the substrate ---------------------------------------------------

    @staticmethod
    def _substrate() -> Any:
        from core.consciousness.steering_channel import her_substrate

        return her_substrate()

    def _state(self, substrate: Any) -> tuple[list[float], list[float]] | None:
        """Her readings and her full state vector, or None when there is no substrate."""
        if substrate is None:
            return None
        x = getattr(substrate, "x", None)
        if x is None:
            return None
        try:
            values = [float(v) for v in list(x)]
        # not a failure: a state that is not numbers gives no reading, the same
        # as no substrate, and the decision goes unscaled.
        except (TypeError, ValueError):
            return None
        readings: list[float] = []
        for name in READINGS:
            index = getattr(substrate, name, None)
            if not isinstance(index, int) or not 0 <= index < len(values):
                return None
            readings.append(values[index])
        return readings, values

    def _free_units(self, substrate: Any, size: int) -> list[int]:
        """The units no reading is taken from, in order: the ones a gate may wander by."""
        taken = {getattr(substrate, name, None) for name in READINGS}
        return [i for i in range(size) if i not in taken]

    def _gate(self, name: str, substrate: Any, size: int) -> _Multiplier | None:
        gate = self._gates.get(name)
        if gate is not None:
            return gate
        free = self._free_units(substrate, size)
        used = {g.unit for g in self._gates.values()}
        spare = [u for u in free if u not in used]
        if not spare:
            return None
        gate = _Multiplier(unit=spare[0])
        self._gates[name] = gate
        return gate

    # -- a decision asks -------------------------------------------------

    def multiplier(self, name: str, *, substrate: Any = None) -> float:
        """What this decision's input is scaled by, now. 1.0 with no substrate or when switched off."""
        if disabled():
            return 1.0
        substrate = substrate if substrate is not None else self._substrate()
        read = self._state(substrate)
        if read is None:
            return 1.0
        readings, values = read
        gate = self._gate(name, substrate, len(values))
        if gate is None:
            return 1.0
        # The shared readings move once per change of her state, however many
        # gates are consulted in between.
        step = getattr(substrate, "_state_revision", None)
        if step is None or step != self._seen_step:
            for running, value in zip(self._readings, readings, strict=True):
                running.add(value)
            self._seen_step = step
        f = [running.standard(value) for running, value in zip(self._readings, readings, strict=True)]
        unit = self._units.setdefault(gate.unit, _Running())
        unit.add(values[gate.unit])
        xi = values[gate.unit] - unit.mean
        drive = sum(w * v for w, v in zip(gate.weights, f, strict=True)) + xi
        for i, v in enumerate(f):
            gate.eligibility[i] += xi * v
        gate.consulted += 1
        gate.last = math.exp(max(-_EXP_LIMIT, min(_EXP_LIMIT, drive)))
        return gate.last

    # -- the turn ends ----------------------------------------------------

    def learn(self, dose: float) -> dict[str, Any]:
        """Move each gate consulted this turn towards the turn's dose times its eligibility."""
        taught: dict[str, Any] = {}
        if disabled():
            return taught
        for name, gate in self._gates.items():
            if gate.consulted == 0:
                continue
            gate.turns += 1
            rate = _rate(gate.turns)
            for i in range(len(READINGS)):
                target = float(dose) * gate.eligibility[i] / gate.consulted
                gate.weights[i] += rate * (target - gate.weights[i])
            taught[name] = gate.consulted
            gate.eligibility = [0.0] * len(READINGS)
            gate.consulted = 0
        return taught

    # -- what is read of it ------------------------------------------------

    def status(self) -> dict[str, Any]:
        return {
            name: {
                "unit": gate.unit,
                "turns": gate.turns,
                "last": gate.last,
                "weights": dict(zip(READINGS, gate.weights, strict=True)),
            }
            for name, gate in sorted(self._gates.items())
        }


def get_substrate_gates() -> SubstrateGates:
    return _GATES


def reset_for_test() -> None:
    global _GATES
    _GATES = SubstrateGates()


#: Made at import, so the subject-core fork carries it; see core/soma/good_news.py.
_GATES: SubstrateGates = SubstrateGates()
#: What each gate has learned is part of her history.
keep_across_stages(__name__, "_GATES")


__all__ = [
    "DISABLE_ENV",
    "FIXED_GATES",
    "READINGS",
    "SubstrateGates",
    "disabled",
    "get_substrate_gates",
    "reset_for_test",
]
