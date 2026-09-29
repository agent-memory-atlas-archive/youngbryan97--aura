"""One row of the null table per synthetic architecture, computed apart from the others.

Each null is a toy system built from its own seed: its draws of irreducibility,
its graph, its closure, its differentiation, intrinsic gain, both synergy suites
and its spread. None of it touches the organism or any other null, so the rows
can be computed in parallel processes and come out the same numbers in the same
order. The null stage of a seed-7 campaign took 4 h 35 min in one process
(`whole-s7-27dc1dda9`, 26 September).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

__all__ = ["null_row", "null_rows"]


def null_row(
    name: str, *, seed: int, null_draws: int, trials: int, domains: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    """One architecture's row, and anything worth logging about how it was measured."""
    from core.subject.graph import analyse_graph
    from core.subject.irreducibility import phi_do
    from core.subject.nulls import architecture, toy_edges, toy_recording

    notes: list[str] = []
    # Several instantiations of each, so a null is a distribution rather than
    # one draw of a random weight matrix. A single instantiation can be lucky in
    # either direction and the comparison is with its tail.
    values: list[float] = []
    edges: list[Any] = []
    graph: Any = None
    for draw in range(null_draws):
        system = architecture(name, seed=seed + draw)
        toy = toy_recording(system, steps=2500, seed=seed + draw)
        values.append(round(phi_do(toy).phi, 5))
        if draw == 0:
            edges = toy_edges(system, trials=trials, seed=seed)
            graph = analyse_graph(list(domains), edges)
    # Whether the null's own core is closed. A broker outside K makes every
    # domain depend on every other one, and what separates it from a mind is
    # that K's future depends on a variable no reading of K contains. A toy's
    # state can be listed, so `toy_closure` reads "nothing outside K" off it.
    closed = True
    leak = 0.0
    try:
        from core.subject.nulls import toy_closure

        closed, leak = toy_closure(architecture(name, seed=seed), steps=2500, seed=seed)
    except (ImportError, ValueError, RuntimeError, AttributeError) as exc:
        notes.append(f"  closure unavailable for the {name} null: {exc}")
    # And the rest of the suite, on the same toy recording, because the nulls
    # are the systems built to fake the other lines of the conjunction too.
    extra: dict[str, Any] = {}
    try:
        from core.subject.differentiation import effective_dimension
        from core.subject.intrinsic import intrinsic_gain
        from core.subject.synergy import synergy_suite

        scored = toy_recording(architecture(name, seed=seed), steps=2500, seed=seed)
        spectrum = effective_dimension(scored)
        extra["d_eff"] = round(float(spectrum.d_eff), 4)
        extra["d_eff_normalised"] = round(float(spectrum.normalised), 4)
        extra["largest_component_share"] = round(float(spectrum.top_share), 4)
        extra["intrinsic_gain"] = round(float(intrinsic_gain(scored, seed=seed).gain), 5)
        reports = synergy_suite(scored, seed=seed)
        extra["synergy"] = [round(float(r.normalised), 4) for r in reports]
        extra["synergy_passes"] = [bool(r.passes) for r in reports]
        change = synergy_suite(scored, seed=seed, of="change")
        extra["synergy_v2_passes"] = [bool(r.passes) for r in change]
        extra["synergy_v3_passes"] = [bool(r.passes_v3) for r in change]
        extra["synergy_min"] = round(min(float(r.normalised) for r in reports), 4) if reports else 0.0
    except (ImportError, ValueError, RuntimeError, AttributeError, TypeError) as exc:
        notes.append(f"  the wider suite was unavailable for the {name} null: {exc}")
    # Spread in its own block: the share of the other domains a source reaches,
    # at the source that reaches most. It needs nothing the lines above produce.
    reached: dict[str, set[str]] = {}
    for source, target in edges:
        reached.setdefault(source, set()).add(target)
    extra["spread"] = round(max((len(v) for v in reached.values()), default=0) / max(1, len(domains) - 1), 4)
    row = {
        "phi_do": round(float(np.quantile(values, 0.95)), 5),
        "draws": values,
        "max": max(values),
        "median": round(float(np.median(values)), 5),
        "quantile": 0.95,
        "kind": "architecture",
        "one_component": graph.one_component,
        "vertex_connectivity": graph.connectivity,
        "reentry": graph.every_node_reenters,
        "closed": closed,
        "leak": round(leak, 5),
        **extra,
    }
    return row, notes


def _one(task: tuple[str, int, int, int, tuple[str, ...]]) -> tuple[str, dict[str, Any], list[str]]:
    name, seed, null_draws, trials, domains = task
    row, notes = null_row(name, seed=seed, null_draws=null_draws, trials=trials, domains=domains)
    return name, row, notes


def null_rows(
    names: Sequence[str],
    *,
    seed: int,
    null_draws: int,
    trials: int,
    domains: Sequence[str],
    workers: int = 1,
) -> list[tuple[str, dict[str, Any], list[str]]]:
    """Every architecture's row, in the order given, across `workers` processes.

    A worker is spawned rather than forked, so it starts from nothing the
    campaign's organism holds, and it reads its numeric thread cap from the
    environment the campaign exported.
    """
    tasks = [(str(name), int(seed), int(null_draws), int(trials), tuple(domains)) for name in names]
    if workers <= 1 or len(tasks) <= 1:
        return [_one(task) for task in tasks]
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks), os.cpu_count() or 1), mp_context=context) as pool:
        return list(pool.map(_one, tasks))
