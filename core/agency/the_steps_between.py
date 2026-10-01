"""How far apart two amounts are, counted in the steps this world takes.

Her judgement of a position counted every gap between amounts in doublings:
nearness to a number she was asked to reach, and how close neighbouring things
are. The reason given was that "in anything built by combining a step is a
doubling", which is true of 2048 and of little else. Threes climbs 1, 2, 3, 6,
12; a merge game on Fibonacci numbers climbs by the golden ratio; a world that
adds one thing at a time climbs by one. Counted in doublings, each of those
is measured in a unit it does not have, and the 2048 prior sits inside code
that calls itself general.

So the step is read off the amounts the world has shown. If the ratios
between consecutive amounts are steadier than the differences, the world
grows by multiplying, and a step is the smallest ratio seen; otherwise it
grows by adding, and a step is the smallest difference seen. Steadier means a
smaller spread relative to its mean, so the two are compared in their own
terms with no threshold chosen for them. A place on the ladder is the number
of steps above the lowest amount seen. On a 2048 board that is exactly the
count of doublings, learned rather than assumed.

The amounts come from the situation being judged, and during a search from
every situation the search has judged so far (`in_view`), so a step her model
predicts is a step she can measure by. Nothing is kept between searches.
"""
from __future__ import annotations

import contextlib
import functools
import math
from collections.abc import Callable, Iterable, Iterator
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, TypeVar

__all__ = ["Ladder", "in_view", "ladder_here", "ladder_of", "within_one_view"]

T = TypeVar("T")

#: The amounts a search has seen, while one is running.
_IN_VIEW: ContextVar[set[float] | None] = ContextVar("aura_amounts_in_view", default=None)


@dataclass(frozen=True)
class Ladder:
    """The steps one world takes between its amounts."""

    lowest: float
    #: The size of one step: a ratio's logarithm when ``multiplies``, else a difference.
    step: float
    multiplies: bool

    def place(self, amount: float) -> float:
        """How many steps this amount is from nothing; never negative.

        Nothing is the world's own identity: one where it multiplies, nought
        where it adds. A 16 is four doublings from one and halfway to 256.
        """
        value = float(amount)
        if self.multiplies:
            if value <= 0.0:
                return 0.0
            return max(0.0, math.log(value) / self.step)
        return max(0.0, value / self.step)

    def apart(self, one: float, other: float) -> float:
        """Steps between two amounts."""
        return abs(self.place(one) - self.place(other))


def _spread(values: list[float]) -> float:
    """Standard deviation over mean; zero when they are all alike."""
    mean = sum(values) / len(values)
    if mean <= 0.0:
        return math.inf
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)) / mean


def ladder_of(amounts: Iterable[float]) -> Ladder | None:
    """The ladder these amounts were climbed on, or None with fewer than two."""
    rungs = sorted({float(a) for a in amounts if a is not None and math.isfinite(float(a))})
    if len(rungs) < 2:
        return None
    differences = [b - a for a, b in zip(rungs, rungs[1:])]
    positive = all(r > 0.0 for r in rungs)
    ratios = [math.log(b / a) for a, b in zip(rungs, rungs[1:])] if positive else []
    if ratios and _spread(ratios) <= _spread(differences):
        return Ladder(lowest=rungs[0], step=min(ratios), multiplies=True)
    return Ladder(lowest=rungs[0], step=min(differences), multiplies=False)


@contextlib.contextmanager
def in_view() -> Iterator[None]:
    """Collect the amounts every judged situation shows, for one search.

    A search that looks deeper by calling itself keeps the view it is in.
    """
    if _IN_VIEW.get() is not None:
        yield
        return
    token = _IN_VIEW.set(set())
    try:
        yield
    finally:
        _IN_VIEW.reset(token)


def within_one_view(search: Callable[..., T]) -> Callable[..., T]:
    """Run ``search`` inside one view of the amounts it judges."""

    @functools.wraps(search)
    def searching(*args: Any, **kwargs: Any) -> T:
        with in_view():
            return search(*args, **kwargs)

    return searching


def ladder_here(amounts: Iterable[float]) -> Ladder | None:
    """The ladder for a situation showing ``amounts``, with what the search has seen."""
    shown = [float(a) for a in amounts if a is not None]
    seen = _IN_VIEW.get()
    if seen is not None:
        seen.update(shown)
        return ladder_of(seen)
    return ladder_of(shown)
