"""How the runtime keeps the topology in step with the population.

Lifted whole out of `runtime`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

from typing import Any


class _KeepsTheTopology:
    """Lifted whole out of MorphogeneticRuntime; see runtime.py."""

    def _sync_topology(self) -> None:
        """Keep the graph's node set equal to the live population.

        Cells are registered and retired by paths outside the governor — boot
        registration, organ formation, quarantine — so the graph follows them
        rather than trying to own them. Bindings are the governor's alone.
        """
        from .runtime import (
            _MORPHOGENESIS_RECOVERABLE_ERRORS,
            _record_morphogenesis_runtime_degradation,
        )

        try:
            live = {cell.cell_id for cell in self.registry.active_cells()}
            known = set(self.graph.nodes())
            if live == known:
                # The population is unchanged, but the bindings may not be:
                # a graph restored from disk arrives with edges the substrate
                # was never told about. Returning here left the two
                # disagreeing on 82 bindings after a restart, which is the
                # signature of a partial failure nobody cleaned up.
                self._attach_the_stranded(live)
                self._reconcile_substrate()
                return
            for cell_id in live - known:
                self.substrate.place(cell_id)
                self.lineage.seed(cell_id, cause="registered")
                cell = self.registry.get(cell_id)
                if cell is not None:
                    self.governor.set_capabilities(cell_id, cell.manifest.capabilities)
            gone = known - live
            arrived = live - known

            # A node synced in with no bindings is its own connected component,
            # so every organ the stabilizer formalizes used to trip the
            # partition alarm on a healthy runtime. Bind an arriving cell to
            # what it is already related to: an organ to its members, anything
            # else to its own subsystem's neighbours.
            attachments = self._attachments_for(arrived, live)

            def sync(scratch: Any) -> None:
                for cell_id in sorted(arrived):
                    scratch.add_node(cell_id)
                for cell_id in sorted(gone):
                    scratch.remove_node(cell_id)
                for edge in attachments:
                    if edge.source not in gone and edge.target not in gone:
                        scratch.add_edge(edge)

            self.graph.transaction(sync, cause=f"population_sync@tick{self._tick}")
            if gone:
                # Whatever the graph recorded about these cells was decided
                # under a world that no longer holds.
                self.governor.invalidate_reversal_history(
                    cells=sorted(gone), reason="cells left the population"
                )
                for cell_id in gone:
                    self.substrate.retire(cell_id)
                    self.lineage.record_retirement(cell_id, cause="left the registry")
            self._reconcile_substrate()
        except _MORPHOGENESIS_RECOVERABLE_ERRORS as exc:
            _record_morphogenesis_runtime_degradation(
                exc,
                action="kept the previous topology after a population sync failed",
                severity="warning",
                extra={"tick": self._tick},
            )

    def _attach_the_stranded(self, live: set[str]) -> None:
        """Bind a live cell the graph holds with no edges at all.

        Attachments were computed only for cells that ARRIVE, and a cell
        already isolated when the graph was saved never arrives again: the
        population matches on every boot, this method returns early, and
        nothing binds it for the life of the installation.

        The persisted graph on 2026-09-20 held 50 nodes, 196 edges and
        exactly six isolated cells — six organs the stabilizer formalised,
        reported live as "population split into 7 pieces: 44,1,1,1,1,1,1"
        and still split after readiness. No degree budget was reached and
        nothing was refused; the healing simply never ran on them.

        Isolation is the condition, not arrival. Anything the attachment
        rule declines — a cell alone in its subsystem, a budget already
        spent — is declined here too, and stays visible in the partition
        count where a policy can decide about it.
        """
        from .runtime import (
            logger,
        )

        stranded = {
            cell_id for cell_id in live
            if not self.graph.out_edges(cell_id) and not self.graph.in_edges(cell_id)
        }
        if not stranded:
            return
        edges = self._attachments_for(stranded, live)
        if not edges:
            return

        def heal(scratch: Any) -> None:
            for edge in edges:
                scratch.add_edge(edge)

        self.graph.transaction(heal, cause=f"attach_stranded@tick{self._tick}")
        logger.info(
            "Morphogenesis bound %d stranded cell(s) the restore left isolated: %s",
            len(stranded),
            ", ".join(sorted(stranded)[:6]),
        )

    def _reconcile_substrate(self) -> None:
        """Make the substrate hold exactly the bindings the graph declares.

        Cheap: a membership test per edge, and work only where they differ.
        Run every tick rather than only when the population changes, because
        the ways they can drift apart — a restored graph, a rolled-back
        commit, a lesion — mostly leave the node set alone.
        """
        from .runtime import (
            EdgeType,
            MorphEdge,
            _record_morphogenesis_runtime_degradation,
        )

        try:
            declared = {edge.key: edge for edge in self.graph.edges()}
            for edge in declared.values():
                if not self.substrate.bound(edge):
                    self.substrate.bind(edge)
            for key in self.substrate.bound_keys() - set(declared):
                self.substrate.unbind(MorphEdge(
                    source=key[0], target=key[1],
                    edge_type=EdgeType(key[2]), port=key[3],
                ))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_morphogenesis_runtime_degradation(
                exc,
                action="left the substrate and the graph disagreeing for one tick",
                severity="warning",
            )

    def _cells_alone_in_their_subsystem(self) -> set[str]:
        """Cells whose subsystem holds only them, which nothing binds by default."""
        by_subsystem: dict[str, list[str]] = {}
        for cell in self.registry.active_cells():
            by_subsystem.setdefault(cell.manifest.subsystem, []).append(cell.cell_id)
        return {
            held[0] for held in by_subsystem.values() if len(held) == 1
        }

    def _attachments_for(self, arrived: set[str], live: set[str]) -> list[Any]:
        """Bindings an arriving cell should come with.

        Only what is structurally true about the cell: an organ names its
        members, so it binds to them; anything else binds to its own
        subsystem's peers.

        A cell that is the only one in a new subsystem gets nothing, and stays
        its own component until something decides to cover it. Binding it to an
        arbitrary peer would make the partition channel quiet and the coverage
        real in neither case — and it would take the decision away from the
        policy, which is the part that has to justify it and can be refused.
        """
        from .graph import EdgeType, MorphEdge
        from .runtime import (
            logger,
        )

        edges: list[Any] = []
        by_subsystem: dict[str, list[str]] = {}
        for cell in self.registry.active_cells():
            by_subsystem.setdefault(cell.manifest.subsystem, []).append(cell.cell_id)

        # The degree budget the transaction will be validated against, counted
        # from what the graph already holds. The reverse edge is what overran
        # it: four bindings per arriving cell is bounded, but the peer on the
        # other end of them is not, so several cells arriving into one
        # subsystem gave a shared peer more outbound edges than the graph
        # allows. The transaction then failed as a whole — node adds, removals
        # and all — and the next tick proposed exactly the same thing, once a
        # second, for the life of the process. A proposer that does not know
        # the constraint that judges it cannot propose something admissible.
        out_degree: dict[str, int] = {}
        in_degree: dict[str, int] = {}
        existing: set[tuple[str, str, Any, str]] = set()
        for edge in self.graph.edges():
            out_degree[edge.source] = out_degree.get(edge.source, 0) + 1
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1
            existing.add((edge.source, edge.target, edge.edge_type, edge.port or ""))
        max_out = self.graph.max_out_degree
        max_in = self.graph.max_in_degree
        # And the envelope the governor holds the whole topology to. This sync
        # is not a proposal, so nothing refused it, and it carried the graph
        # to 258 bindings against a ceiling of 256 on every boot:
        # morphogenesis.edges red_high before a single transition was asked
        # for (2026-09-21, and again 23 Sep).
        room = max(0, int(self.governor.bounds.max_edges) - len(existing))
        clipped = 0

        def admit(source: str, target: str) -> bool:
            nonlocal clipped
            identity = (source, target, EdgeType.OBSERVE, "")
            if identity in existing:
                return False
            if (
                out_degree.get(source, 0) >= max_out
                or in_degree.get(target, 0) >= max_in
                or len(edges) >= room
            ):
                clipped += 1
                return False
            out_degree[source] = out_degree.get(source, 0) + 1
            in_degree[target] = in_degree.get(target, 0) + 1
            existing.add(identity)
            edges.append(MorphEdge(
                source=source, target=target, edge_type=EdgeType.OBSERVE, weight=0.6,
            ))
            return True

        reverse: list[tuple[str, str]] = []
        # Where each arriving cell may bind: its organ's members, or its
        # subsystem's peers where it names none or they have no room left.
        #
        # The fallback to subsystem peers ran only when the member list was
        # EMPTY, so an organ whose members are all at the degree cap got
        # nothing at all and stayed its own component — while peers with room
        # sat beside it. Measured on the live graph (2026-09-20): six organs,
        # every one of their members at exactly 16/16, and 29 of 29 `global`
        # peers under the cap. "Cannot be reached" includes "has no room left"
        # as well as "is not there".
        reaches: dict[str, tuple[list[str], list[str]]] = {}
        for cell_id in sorted(arrived):
            cell = self.registry.get(cell_id)
            if cell is None:
                continue
            peers = [
                peer for peer in by_subsystem.get(cell.manifest.subsystem, ())
                if peer != cell_id and peer in live
            ]
            members = [
                str(m) for m in (cell.manifest.metadata.get("members") or ())
                if str(m) in live and str(m) != cell_id
            ]
            if not members:
                members, peers = peers, []
            reaches[cell_id] = (members, peers)

        def with_the_most_room(names: list[str]) -> list[str]:
            # The peers with the most room, not the first by name. Twenty
            # cells arriving into one subsystem all chose the same
            # alphabetically-first peers, saturated them, and left the last
            # four with no binding at all.
            return sorted(
                names, key=lambda peer: (out_degree.get(peer, 0) + in_degree.get(peer, 0), peer)
            )

        # One binding for every arriving cell before a second for any. The
        # first joins a cell to its component; the rest are redundancy, and
        # where the envelope runs short it is redundancy that goes.
        taken: dict[str, int] = {cell_id: 0 for cell_id in reaches}
        for round_cap in range(1, 5):
            for cell_id, (members, peers) in reaches.items():
                if taken[cell_id] >= round_cap:
                    continue
                choices = with_the_most_room(members)
                for target in choices + (with_the_most_room(peers) if not taken[cell_id] else []):
                    if taken[cell_id] >= round_cap:
                        break
                    # The forward edge is what joins the arriving cell to its
                    # component; the reverse one waits for the end.
                    if admit(cell_id, target):
                        taken[cell_id] += 1
                        reverse.append((target, cell_id))
        # The reverse edges last. Each costs symmetry and not connectivity, so
        # where the envelope runs short it is these that go, never the edge
        # that joins a cell to its component.
        for source, target in reverse:
            admit(source, target)
        if clipped:
            logger.debug(
                "Morphogenesis population sync left %d attachment(s) unbound at the "
                "degree budget (max_out=%d, max_in=%d)", clipped, max_out, max_in,
            )
        return edges

