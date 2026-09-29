"""The graph is recomputed per condition, and one workload cannot stand for her life.

Phase 1 of the ISC completion list asks for every coupling measure to be
recomputed rather than carried over, and Phase 44's criterion is what a
per-condition graph is for: a graph that is strongly connected only while she is
solving a problem is a graph of that workload, not of her.

So the graph is rebuilt for each condition she was recorded under, and
`natural_runtime_replication` reads the count of conditions in which it is
strongly connected. Three is the bar.
"""

from __future__ import annotations

import pytest

from core.subject.battery import assemble

pytestmark = pytest.mark.unit

CONDITIONS = (
    "conversation", "problem_solving", "autonomy", "idle",
    "salience", "stress", "memory", "tool_use",
)


def _evidence(with_scc: int, graphs: int = len(CONDITIONS)) -> dict:
    return {
        "per_condition": {
            "conditions_with_scc": with_scc,
            "per_condition": {
                name: {"one_strongly_connected_component": i < with_scc}
                for i, name in enumerate(CONDITIONS[:graphs])
            },
        }
    }


def _replication(evidence: dict):
    verdict = assemble(evidence)
    return next(c for c in verdict.criteria if c.key == "natural_runtime_replication")


def test_a_graph_that_holds_in_three_conditions_replicates():
    assert _replication(_evidence(3)).passed


def test_one_that_holds_in_two_does_not():
    assert not _replication(_evidence(2)).passed


def test_a_run_with_no_per_condition_graphs_cannot_claim_it():
    assert not _replication({}).passed


def test_the_criterion_carries_the_graphs_it_read():
    criterion = _replication(_evidence(8))
    assert set(criterion.value) == set(CONDITIONS)


def test_the_bar_is_stated_in_the_criterion():
    assert "three" in _replication(_evidence(3)).bar


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
