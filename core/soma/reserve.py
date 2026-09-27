"""What good turns leave her with, to work on.

A cell does not burn its food for work. The food's energy pumps protons across
the inner mitochondrial membrane, and ATP synthase turns the flow back down that
gradient into ATP, the currency most of the cell runs on; with no gradient the
rotor stops and the cell starves (docs/WHAT_A_MITOCHONDRION_TEACHES.md, 1).

She had the spending half. Her exertion drains her energy budget and rest
restores it (`MotivationUpdatePhase._spend_energy`), and fatigue narrows what
she attends to when the spending runs long (core/soma/fatigue.py). Nothing she
did well gave any of it back: a stretch of turns that went better than she
expected left her exactly as spent as a stretch that went worse. People are not
like that. Success is energising and failure draining, and positive affect
restores depleted self-control (Tice, Baumeister, Shmueli and Muraven 2007).

So a turn's worth charges a reserve and the reserve pays for her exertion:

    ordinary  what a turn in which she spent anything usually spent, the median
              of her own recent spending
    charge    a turn better than she expected adds its dose times the ordinary;
              her best turn in a while pays back one ordinary turn's spending
    drain     a turn worse than she expected takes the same measure away
    flow      when she spends, the reserve covers as much as it holds; when she
              is resting nothing is drawn, as a gradient nobody draws on is not
              run down
    ceiling   one full energy budget: what the reserve holds beyond what she
              could ever spend is not held
    leak      a share of it goes each turn, one over the window her other
              ledgers keep, as a membrane leaks protons whether or not they
              are used

The dose is the payoff layer's (core/affect/what_it_was_worth.py), so the
reserve charges only on what was better or worse than expected, and a routine
that always pays stops charging it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from core.self.what_came_before import keep_across_stages

__all__ = ["Reserve", "ReserveLedger", "get_reserve_ledger", "reset_for_test"]

#: The window her other ledgers keep: how many spending turns the ordinary is
#: taken over, and one over it is the share that leaks each turn.
_WINDOW = 256


@dataclass(frozen=True)
class Reserve:
    """The reserve, what an ordinary spending turn costs, and what it has paid for."""

    held: float
    ordinary: float
    paid: float
    charged: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "held": round(self.held, 6),
            "ordinary": round(self.ordinary, 6),
            "paid": round(self.paid, 6),
            "charged": round(self.charged, 6),
        }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


class ReserveLedger:
    """The reserve her good turns charge and her exertion draws on."""

    def __init__(self) -> None:
        self.held = 0.0
        self._spending: deque[float] = deque(maxlen=_WINDOW)
        #: The energy budget's capacity, the reserve's ceiling. Set by the phase
        #: that spends energy, which is the one that knows it.
        self.capacity = 100.0
        self.paid = 0.0
        self.charged = 0.0

    def ordinary(self) -> float:
        """What a turn in which she spent anything usually spent."""
        return _median(list(self._spending))

    def charge(self, amount: float) -> float:
        """A turn's dose, in [-1, 1]: better charges the reserve, worse drains it. Returns the change."""
        try:
            amount = float(amount)
        # not a failure: a dose that is not a number moves nothing.
        except (TypeError, ValueError):
            return 0.0
        if amount != amount:
            return 0.0
        amount = max(-1.0, min(1.0, amount))
        before = self.held
        self.held *= 1.0 - 1.0 / _WINDOW
        self.held = max(0.0, min(self.capacity, self.held + amount * self.ordinary()))
        change = self.held - before
        if change > 0.0:
            self.charged += change
        return change

    def draw(self, spend: float, *, capacity: float | None = None) -> float:
        """Cover as much of a turn's spending as the reserve holds. Returns what it covered.

        Called with what the turn would have taken from her energy. Resting,
        which gives energy back, draws nothing.
        """
        try:
            spend = float(spend)
        # not a failure: a spend that is not a number draws nothing.
        except (TypeError, ValueError):
            return 0.0
        if capacity is not None and capacity > 0.0:
            self.capacity = float(capacity)
            self.held = min(self.held, self.capacity)
        if not spend > 0.0:
            return 0.0
        self._spending.append(spend)
        covered = min(self.held, spend)
        self.held -= covered
        self.paid += covered
        return covered

    def read(self) -> Reserve:
        return Reserve(held=self.held, ordinary=self.ordinary(), paid=self.paid, charged=self.charged)

    def share(self) -> float:
        """The reserve as a share of a full energy budget."""
        return self.held / self.capacity if self.capacity > 0.0 else 0.0


def get_reserve_ledger() -> ReserveLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = ReserveLedger()


#: Made at import rather than on first use, and kept across her restarts along
#: her own line. See core/soma/good_news.py and core/self/what_came_before.py.
_LEDGER: ReserveLedger = ReserveLedger()
keep_across_stages(__name__, "_LEDGER")
