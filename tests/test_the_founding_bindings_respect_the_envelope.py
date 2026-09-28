"""The founding bindings are held to the same envelope as every later one."""
from __future__ import annotations

import types

import pytest

from core.morphogenesis import integration as I
from core.morphogenesis.governor import MorphBounds, MorphGovernor
from core.morphogenesis.graph import MorphGraph
from core.morphogenesis.lineage import Lineage
from core.morphogenesis.registry import MorphogenesisRegistry
from core.morphogenesis.substrate import SimulationSubstrate
from core.morphogenesis.types import CellManifest

pytestmark = pytest.mark.unit

#: The live population's shape, read from the boot line of 2026-09-28:
#: "53 cell(s)" across the subsystems the adjacency table names.
_LIVE_SHAPE = {
    "state": 6, "memory": 7, "cognition": 8, "llm_router": 4, "resilience": 5,
    "homeostasis": 5, "affect": 4, "consciousness": 5, "social": 4, "tools": 5,
}


def _runtime(tmp_path) -> types.SimpleNamespace:
    registry = MorphogenesisRegistry(root=tmp_path)
    for subsystem, count in _LIVE_SHAPE.items():
        for n in range(count):
            registry.register_cell(
                CellManifest(name=f"{subsystem}_{n}", subsystem=subsystem)
            )
    graph = MorphGraph()
    governor = MorphGovernor(
        graph=graph, substrate=SimulationSubstrate(),
        lineage=Lineage(), bounds=MorphBounds(max_cells=128),
        require_governance=False,
    )
    return types.SimpleNamespace(
        config=types.SimpleNamespace(topology_enabled=True),
        registry=registry, graph=graph, governor=governor,
        substrate=governor.substrate, lineage=governor.lineage,
    )


def test_the_boot_seed_stays_inside_the_governors_envelope(tmp_path):
    """LIVE 21, 23 and 28 September: 258 bindings against a ceiling of 256.

    `_seed_topology` added every adjacency pair in both directions and was
    checked against the graph's own degree caps only, so `morphogenesis.edges`
    reported red_high from the first tick of every process — before a single
    transition had been proposed — and the governor's `max_edges` was a bound
    the topology had already passed on arrival.
    """
    rt = _runtime(tmp_path)
    I._seed_topology(rt)
    assert rt.graph.edge_count <= rt.governor.bounds.max_edges, (
        f"{rt.graph.edge_count} bindings seeded against a ceiling of "
        f"{rt.governor.bounds.max_edges}"
    )


def test_the_clip_is_actually_reached_by_this_population(tmp_path):
    """A test that never crosses the line proves nothing about the line."""
    rt = _runtime(tmp_path)
    pairs = 0
    by_subsystem: dict[str, list[str]] = {}
    for cell in rt.registry.active_cells():
        by_subsystem.setdefault(cell.manifest.subsystem, []).append(cell.cell_id)
    for left, right, _weight in I._TISSUE_ADJACENCY:
        for source in by_subsystem.get(left, ()):
            for target in by_subsystem.get(right, ()):
                if source != target:
                    pairs += 2
    assert pairs > rt.governor.bounds.max_edges, (
        f"this population asks for {pairs} bindings, which does not reach the "
        f"ceiling of {rt.governor.bounds.max_edges}"
    )


def test_what_goes_is_symmetry_and_not_connectivity(tmp_path):
    """Forward edges join subsystems; reverse ones only make them mutual.

    The sync clips in that order for that reason, and the seed now does too. A
    seed that dropped forward edges would leave a subsystem its own component
    on a healthy boot, which is the alarm this seed exists to prevent.
    """
    rt = _runtime(tmp_path)
    I._seed_topology(rt)
    edges = list(rt.graph.edges())
    forward = {(e.source, e.target) for e in edges}
    mutual = sum(1 for e in edges if (e.target, e.source) in forward)
    assert len(rt.graph.components()) == 1, (
        f"the seed left {len(rt.graph.components())} component(s)"
    )
    assert mutual < len(edges), "nothing was clipped, so the order was never tested"


def test_a_seed_the_validator_would_refuse_is_never_proposed(tmp_path):
    """A transaction that passes a budget changes nothing at all.

    `rt.graph.transaction` validates the whole scratch and raises on a degree
    overrun, and `_seed_topology` catches that as a wiring failure — so a
    population shaped to press the per-cell cap got NO founding bindings, every
    cell its own component and a MARGINAL fault, which is the alarm the seed
    exists to prevent. Measured at in-degree 22 against a cap of 16 while the
    clip order was being fixed.
    """
    rt = _runtime(tmp_path)
    for n in range(40):
        rt.registry.register_cell(
            CellManifest(name=f"crowd_{n}", subsystem="state")
        )
    I._seed_topology(rt)
    assert rt.graph.edge_count > 0, "the seed committed nothing"
    out_degree: dict[str, int] = {}
    in_degree: dict[str, int] = {}
    for edge in rt.graph.edges():
        out_degree[edge.source] = out_degree.get(edge.source, 0) + 1
        in_degree[edge.target] = in_degree.get(edge.target, 0) + 1
    assert max(out_degree.values()) <= rt.graph.max_out_degree
    assert max(in_degree.values()) <= rt.graph.max_in_degree
    assert rt.graph.edge_count <= rt.governor.bounds.max_edges
