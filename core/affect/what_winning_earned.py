"""What each source that won her attention has earned from the turns it won.

The workspace picks one winner a tick, and a turn is many ticks. When the turn
ends and its worth is known (core/affect/what_it_was_worth.py), each source that
won during it is credited with that worth in proportion to how much of the turn
it held. What a source has earned is the running mean of the worth of the turns
it was credited in: how much better or worse than she expected things went when
that source had her attention. It is the actor's half of an actor-critic (Barto,
Sutton and Anderson 1983). The worth is the critic's error, and the sources
bidding for her attention are the actions it teaches.

What a source has earned multiplies its next bids in
core/consciousness/global_workspace.py, beside what she has come to feel about
it (core/affect/feelings_about.py), and is scaled the way that ledger scales its
pull, by how established it is, seen / (seen + 1). A source that keeps winning
turns that go well bids stronger. One that keeps winning turns that go badly
bids weaker and loses her attention to the others.

A source is judged against what turns that started the same way usually bring,
the state-value baseline of an actor-critic (Sutton and Barto 2018, 13.4).
Without it, whatever wins her attention when things are going badly is blamed
for them: an alarm that is right more often is heard less. The baseline is a
straight line of the turn's dose on the valence she arrived with, fitted over
her last 256 turns, and a source earns what its turns brought beyond it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any

import numpy as np

from core.self.what_came_before import keep_across_stages

__all__ = ["CreditLedger", "get_credit_ledger", "reset_for_test"]

#: How many turns of credit a source's earnings are averaged over; the window
#: every other reading of her own history uses.
_WINDOW = 256


class CreditLedger:
    """Each source's earnings, and the workspace's win counts at the last turn's end."""

    def __init__(self) -> None:
        self._wins_before: dict[str, int] = {}
        self._earned: dict[str, float] = {}
        self._weight: dict[str, float] = {}
        self._seen: dict[str, int] = {}
        self.turns = 0
        #: (valence she arrived with, the turn's dose) for her recent measured turns.
        self._outcomes: deque[tuple[float, float]] = deque(maxlen=_WINDOW)
        #: The organs that supplied the winning bids, credited the same way by
        #: their share of what the bids were made of, and kept apart from the
        #: sources: see core/consciousness/global_workspace_supply.py.
        self._supplied_before: dict[str, float] = {}
        self._supplier_earned: dict[str, float] = {}
        self._supplier_weight: dict[str, float] = {}
        self._supplier_seen: dict[str, int] = {}

    def baseline(self, arrived_with: float) -> float:
        """What a turn that started at this valence has usually brought, from her own recent turns."""
        if not self._outcomes:
            return 0.0
        rows = np.asarray(self._outcomes, dtype=np.float64)
        start, brought = rows[:, 0], rows[:, 1]
        spread = float(np.var(start))
        if len(rows) < 3 or spread <= 1e-12:
            return float(np.mean(brought))
        slope = float(np.mean((start - start.mean()) * (brought - brought.mean()))) / spread
        return float(brought.mean() + slope * (arrived_with - start.mean()))

    def note(
        self,
        wins: Mapping[str, int],
        amount: float,
        *,
        measured: bool = True,
        arrived_with: float = 0.0,
        supplied: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Credit this turn's winners with its worth, by their share of the turn's wins.

        `supplied` is the workspace's running total of how much of the winning
        bids each organ supplied; each organ that supplied during the turn is
        credited with the same worth by its share of that supply.

        `wins` is the workspace's running count of wins by source. `amount` is
        the turn's worth as a signed dose in [-1, 1]. A turn whose worth was not
        measured moves the count on and credits nobody: what it was worth is
        unknown, which is not the same as nothing. Returns each winner's share.
        """
        counts: dict[str, int] = {}
        for name, count in wins.items():
            try:
                counts[str(name)] = int(count)
            # not a failure: a count that is not a number is no wins this turn.
            except (TypeError, ValueError):
                continue
        gained = {
            name: count - self._wins_before.get(name, 0)
            for name, count in counts.items()
            if count > self._wins_before.get(name, 0)
        }
        self._wins_before = counts
        supply_gained = self._supply_gained(supplied)
        total = sum(gained.values())
        if not measured or total <= 0:
            return {}
        try:
            amount = float(amount)
        # not a failure: an amount that is not a number credits as neutral.
        except (TypeError, ValueError):
            amount = 0.0
        # Before the clamp: min(1.0, nan) is 1.0, the best turn she could have.
        if amount != amount:
            amount = 0.0
        amount = max(-1.0, min(1.0, amount))
        try:
            arrived_with = float(arrived_with)
        # not a failure: a start that is not a number is judged against every turn.
        except (TypeError, ValueError):
            arrived_with = 0.0
        if arrived_with != arrived_with:
            arrived_with = 0.0
        # Judged against the baseline before this turn joins it.
        advantage = max(-1.0, min(1.0, amount - self.baseline(arrived_with)))
        self._outcomes.append((arrived_with, amount))
        shares = {name: count / total for name, count in gained.items()}
        for name, share in shares.items():
            self._seen[name] = self._seen.get(name, 0) + 1
            self._weight[name] = self._weight.get(name, 0.0) + share
            before = self._earned.get(name, 0.0)
            rate = share / min(self._weight[name], float(_WINDOW))
            self._earned[name] = before + (advantage - before) * rate
        supply_total = sum(supply_gained.values())
        for name, amount_supplied in supply_gained.items():
            share = amount_supplied / supply_total
            self._supplier_seen[name] = self._supplier_seen.get(name, 0) + 1
            self._supplier_weight[name] = self._supplier_weight.get(name, 0.0) + share
            before = self._supplier_earned.get(name, 0.0)
            rate = share / min(self._supplier_weight[name], float(_WINDOW))
            self._supplier_earned[name] = before + (advantage - before) * rate
        self.turns += 1
        return shares

    def _supply_gained(self, supplied: Mapping[str, float] | None) -> dict[str, float]:
        """How much each organ supplied since the last turn's end, and move the mark on."""
        if supplied is None:
            return {}
        totals: dict[str, float] = {}
        for name, value in supplied.items():
            try:
                number = float(value)
            # not a failure: a total that is not a number is no supply this turn.
            except (TypeError, ValueError):
                continue
            if number == number:
                totals[str(name)] = number
        gained = {
            name: value - self._supplied_before.get(name, 0.0)
            for name, value in totals.items()
            if value > self._supplied_before.get(name, 0.0)
        }
        self._supplied_before = totals
        return gained

    def supplier_earned(self, organ: str) -> float:
        """What supplying winning bids has earned this organ, in [-1, 1], scaled by how established it is."""
        seen = self._supplier_seen.get(organ, 0)
        if not seen:
            return 0.0
        return self._supplier_earned.get(organ, 0.0) * seen / (seen + 1)

    def earned(self, source: str) -> float:
        """What winning has earned this source, in [-1, 1], scaled by how established it is."""
        seen = self._seen.get(source, 0)
        if not seen:
            return 0.0
        return self._earned.get(source, 0.0) * seen / (seen + 1)

    def read(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "sources": {
                name: {"earned": round(self.earned(name), 6), "seen": self._seen[name]}
                for name in sorted(self._seen)
            },
            "suppliers": {
                name: {"earned": round(self.supplier_earned(name), 6), "seen": self._supplier_seen[name]}
                for name in sorted(self._supplier_seen)
            },
        }


def get_credit_ledger() -> CreditLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = CreditLedger()


#: Made at import rather than on first use, and kept across her restarts along
#: her own line. See core/soma/good_news.py and core/self/what_came_before.py.
_LEDGER: CreditLedger = CreditLedger()
keep_across_stages(__name__, "_LEDGER")
