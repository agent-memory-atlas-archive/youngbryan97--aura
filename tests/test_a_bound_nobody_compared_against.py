"""The governor's edge bound is enforced, not merely declared.

LIVE, 2026-09-21, in the verifier's own output:

    🔎 VERIFIER [error] telemetry.no_channel_is_red @ morphogenesis.edges:
       morphogenesis.edges is red_high at 258count and has been for 2302s

258 bindings against ``MorphBounds.max_edges = 256``. The bound had been
there from the first commit of the layer and nothing ever compared anything
to it: the governor capped the population against ``population_delta`` and
there was no ``binding_delta`` to cap edges against. The telemetry channel
was the only thing that noticed, for thirty-eight minutes, while the
governor went on admitting binds.
"""

from __future__ import annotations

from core.morphogenesis.graph import MorphEdge
from core.morphogenesis.proposal import (
    MorphProposal,
    MorphTransition,
    TransitionKind,
)


def _bind(source: str, target: str) -> MorphTransition:
    return MorphTransition(
        kind=TransitionKind.BIND,
        subject=source,
        edge=MorphEdge(source=source, target=target),
    )


def test_binding_delta_counts_binds_and_unbinds():
    from core.morphogenesis.proposal import MorphProposal as P

    grew = P(proposer="a", transitions=(_bind("a", "b"), _bind("a", "c")))
    assert grew.binding_delta == 2

    mixed = P(
        proposer="a",
        transitions=(
            _bind("a", "b"),
            MorphTransition(
                kind=TransitionKind.UNBIND,
                subject="a",
                edge=MorphEdge(source="a", target="c"),
            ),
        ),
    )
    assert mixed.binding_delta == 0


def test_a_route_does_not_change_the_count():
    """ROUTE changes a weight, not an edge's existence."""
    routed = MorphProposal(
        proposer="a",
        transitions=(
            MorphTransition(
                kind=TransitionKind.ROUTE,
                subject="a",
                edge=MorphEdge(source="a", target="b"),
                weight=0.5,
            ),
        ),
    )
    assert routed.binding_delta == 0


def _governed(max_edges: int):
    """Three cells joined a-b-c, under a governor whose edge cap is `max_edges`."""
    from core.morphogenesis.governor import MorphBounds, MorphGovernor
    from core.morphogenesis.graph import EdgeType, MorphGraph
    from core.morphogenesis.substrate import SimulationSubstrate

    graph = MorphGraph()

    def build(scratch):
        for node in ("a", "b", "c"):
            scratch.add_node(node)
        scratch.add_edge(MorphEdge("a", "b", EdgeType.DATA, port="x"))
        scratch.add_edge(MorphEdge("b", "c", EdgeType.DATA, port="y"))

    graph.transaction(build, cause="test")
    substrate = SimulationSubstrate(seed=7)
    for node in graph.nodes():
        substrate.place(node)
    governor = MorphGovernor(
        graph,
        substrate,
        bounds=MorphBounds(cooldown_s=0.0, max_edges=max_edges),
        emit_receipts=False,
        shadow_evaluator=lambda graph, proposal=None: 1.0,
    )
    for node in graph.nodes():
        governor.lineage.seed(node)
    governor.credit("a", 50.0)
    return graph, governor


def test_the_governor_refuses_a_bind_past_the_bound():
    """The claim the telemetry channel was making alone.

    Two edges against a cap of two: a third bind is refused on the bound and
    the graph keeps its two.
    """
    from core.morphogenesis import proposal
    from core.morphogenesis.proposal import Decision

    graph, governor = _governed(max_edges=2)
    transaction = governor.adjudicate(proposal.bind("a", "c", "x", proposer="a", benefit=0.5))
    assert transaction.decision is Decision.REJECTED
    assert "bindings: 3 edges would pass the cap of 2" in transaction.reason
    assert graph.edge_count == 2


def test_the_same_bind_under_a_higher_bound_is_not_refused_for_it():
    """The refusal above is the bound's, not some other rung's."""
    from core.morphogenesis import proposal

    _graph, governor = _governed(max_edges=3)
    transaction = governor.adjudicate(proposal.bind("a", "c", "x", proposer="a", benefit=0.5))
    assert "bindings" not in transaction.reason


def test_an_unbind_at_the_bound_is_not_refused_for_it():
    """`binding_delta` is what is compared, so a removal is always in bounds."""
    from core.morphogenesis import proposal

    graph, governor = _governed(max_edges=2)
    edge = next(e for e in graph.edges() if e.source == "a")
    reason = governor._check_bounds(proposal.unbind(edge, proposer="a", benefit=1.0), now=0.0)
    assert not reason.startswith("bindings")


def test_the_bound_and_the_channel_agree():
    """A red channel and an admitted proposal cannot both be right."""
    from core.morphogenesis.governor import MorphBounds
    from core.morphogenesis.telemetry import CHANNEL_EDGES

    from core.fsw.telemetry_dictionary import get_telemetry
    from core.morphogenesis.telemetry import declare

    declare()
    channel = get_telemetry().spec(CHANNEL_EDGES)
    assert channel is not None, "the edges channel must be declared"
    assert int(channel.limits.red_high) == MorphBounds().max_edges, (
        "the channel turns red at the count the governor permits, so one of "
        "them is wrong"
    )
