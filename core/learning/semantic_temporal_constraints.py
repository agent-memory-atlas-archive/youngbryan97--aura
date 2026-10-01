"""Exact difference constraints for interpreted event time.

Observation availability is a separate clock. Constraints can describe past or
future events, but only evidence already available in the requested world is
used. Feasibility certifies the stated constraints, not the source's truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

import networkx as nx

from core.verify.invariants import invariant


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError("temporal bounds require exact integers or rational strings")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rational temporal bound") from exc


@dataclass(frozen=True)
class TemporalConstraint:
    """lower <= time(right) - time(left) <= upper, in one declared clock."""

    left: str
    right: str
    lower: Fraction | None
    upper: Fraction | None
    source: str
    available_at: int = 0
    world: str = "actual"

    def __post_init__(self):
        if (any(not isinstance(v, str) or not v for v in
                (self.left, self.right, self.source, self.world))
                or type(self.available_at) is not int or self.available_at < 0
                or self.lower is None and self.upper is None):
            raise ValueError("temporal constraint needs endpoints, scope and evidence")
        for name in ("lower", "upper"):
            if getattr(self, name) is not None:
                object.__setattr__(self, name, rational(getattr(self, name)))
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("temporal interval is empty")


@dataclass(frozen=True)
class TemporalAssessment:
    consistent: bool
    world: str
    cutoff: int
    bounds: object
    evidence: tuple[str, ...]
    excluded: tuple[str, ...]

    def interval(self, left, right):
        if not self.consistent:
            raise ValueError("inconsistent temporal premises cannot support an inference")
        if left not in self.bounds or right not in self.bounds:
            raise ValueError("temporal endpoint is absent")
        upper, reverse = self.bounds[left][right], self.bounds[right][left]
        return (None if reverse is None else -reverse), upper

    def entails(self, constraint):
        if constraint.world != self.world:
            return False
        lower, upper = self.interval(constraint.left, constraint.right)
        return ((constraint.lower is None or lower is not None and lower >= constraint.lower)
                and (constraint.upper is None or upper is not None and upper <= constraint.upper))


def assess_temporal_constraints(constraints, *, cutoff, world="actual", points=()):
    """Use the installed NetworkX shortest-path solver with rational weights."""
    constraints, points = tuple(constraints), tuple(points)
    if (type(cutoff) is not int or cutoff < 0 or not isinstance(world, str) or not world
            or len(constraints) > 2048 or any(not isinstance(c, TemporalConstraint) for c in constraints)
            or any(not isinstance(p, str) or not p for p in points)):
        raise ValueError("invalid bounded temporal problem")
    admitted = tuple(c for c in constraints if c.world == world and c.available_at <= cutoff)
    nodes = set(points) | {p for c in admitted for p in (c.left, c.right)}
    if len(nodes) > 128:
        raise ValueError("temporal closure exceeds its declared point bound")
    graph = nx.DiGraph()
    graph.add_nodes_from(sorted(nodes))
    for c in admitted:
        for left, right, weight in ((c.left, c.right, c.upper),
                                     (c.right, c.left, None if c.lower is None else -c.lower)):
            if weight is not None and (not graph.has_edge(left, right)
                                      or graph[left][right]["weight"] > weight):
                graph.add_edge(left, right, weight=weight)
    try:
        distances = dict(nx.floyd_warshall(graph))
        consistent = all(distances[p][p] >= 0 for p in nodes)
    except nx.NetworkXUnbounded:
        distances, consistent = {}, False
    bounds = {p: MappingProxyType({q: None if distances[p][q] == float("inf")
                                 else Fraction(distances[p][q]) for q in sorted(nodes)})
              for p in sorted(nodes)} if consistent else {}
    return TemporalAssessment(consistent, world, cutoff, MappingProxyType(bounds),
        tuple(sorted({c.source for c in admitted})),
        tuple(sorted({c.source for c in constraints if c not in admitted})))


@invariant("learning.event_time_does_not_grant_future_evidence", scope="learning",
           owner="core/learning/semantic_temporal_constraints.py", observational=False)
def temporal_custody_contract():
    constraints = (TemporalConstraint("a", "b", 2, 3, "seen", 1),
                   TemporalConstraint("a", "b", -3, -2, "future_observation", 5))
    early = assess_temporal_constraints(constraints, cutoff=1)
    assert early.interval("a", "b") == (2, 3)
    assert not assess_temporal_constraints(constraints, cutoff=5).consistent
    return ()
