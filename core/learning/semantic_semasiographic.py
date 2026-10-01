"""Meaning diagrams independent of pronunciation, layout and reading order.

Arbitrary predicates and named roles can be represented. Only frames with a
declared typed floor interpretation can be executed. Perception or proposal
code supplies symbol occurrences; this module does not infer a glyph grammar
from pixels or confer truth on an interpretation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import MappingProxyType

import networkx as nx

from core.cognition.structure_mapping import Graph, Relation
from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent
from core.learning.semantic_program_floor import (
    compile_source_independent_program_to_floor,
    execute_semantic_floor_program,
    semantic_primitive_type_signature,
)
from core.learning.semantic_program_ir import normalize_semantic_value
from core.learning.semantic_temporal_constraints import (
    TemporalConstraint,
    assess_temporal_constraints,
)
from core.verify.invariants import invariant


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False, ensure_ascii=True).encode("ascii")).hexdigest()


def source_key(key):
    if (not isinstance(key, tuple) or len(key) != 2
            or any(not isinstance(v, str) or not v for v in key)):
        raise ValueError("meaning needs a namespace-qualified source identity")
    return key


def address(key):
    return json.dumps(source_key(key), separators=(",", ":"))


@dataclass(frozen=True)
class DiagramReferent:
    key: tuple[str, str]
    type_name: str
    source: str
    available_at: int = 0
    world: str = "actual"

    def __post_init__(self):
        source_key(self.key)
        if (any(not isinstance(v, str) or not v for v in (self.type_name, self.source, self.world))
                or type(self.available_at) is not int or self.available_at < 0):
            raise ValueError("invalid diagram referent evidence")


@dataclass(frozen=True)
class DiagramEvent:
    key: tuple[str, str]
    symbol: str
    ports: tuple[tuple[str, tuple[str, str] | None], ...]
    source: str
    available_at: int = 0
    world: str = "actual"
    polarity: bool = True
    modality: str = "asserted"

    def __post_init__(self):
        source_key(self.key)
        ports = tuple(self.ports)
        if (any(not isinstance(v, str) or not v for v in (self.symbol, self.source, self.world, self.modality))
                or type(self.available_at) is not int or self.available_at < 0
                or type(self.polarity) is not bool or not ports or len(ports) > 32
                or len({r for r, _ in ports}) != len(ports)
                or any(not isinstance(r, str) or not r for r, _ in ports)):
            raise ValueError("invalid meaning event or named ports")
        for _, key in ports:
            if key is not None:
                source_key(key)
        object.__setattr__(self, "ports", tuple(sorted(ports)))


@dataclass(frozen=True)
class MeaningDiagram:
    referents: tuple[DiagramReferent, ...]
    events: tuple[DiagramEvent, ...]
    temporal: tuple = ()

    def __post_init__(self):
        refs, events = tuple(self.referents), tuple(self.events)
        if (any(not isinstance(v, DiagramReferent) for v in refs)
                or any(not isinstance(v, DiagramEvent) for v in events)
                or any(not isinstance(v, TemporalConstraint) for v in self.temporal)):
            raise ValueError("meaning diagram needs typed source records")
        keys = {v.key for v in (*refs, *events)}
        if (not events or len(refs) + len(events) > 128 or len(keys) != len(refs) + len(events)
                or any(k is not None and k not in keys for e in events for _, k in e.ports)):
            raise ValueError("meaning diagram has duplicate or absent identities")
        object.__setattr__(self, "referents", tuple(sorted(refs, key=lambda v: v.key)))
        object.__setattr__(self, "events", tuple(sorted(events, key=lambda v: v.key)))
        object.__setattr__(self, "temporal", tuple(self.temporal))

    @property
    def meaning_sha256(self):
        return digest({"referents": [(v.key, v.type_name, v.world) for v in self.referents],
            "events": [(e.key, e.symbol, e.ports, e.world, e.polarity, e.modality) for e in self.events],
            "time": sorted((c.left, c.right, str(c.lower), str(c.upper), c.world) for c in self.temporal)})

    @property
    def custody_sha256(self):
        return digest({"meaning": self.meaning_sha256,
            "witnesses": [(v.key, v.source, v.available_at) for v in (*self.referents, *self.events)],
            "time": sorted((c.left, c.right, str(c.lower), str(c.upper), c.world,
                            c.source, c.available_at) for c in self.temporal)})

    def structure_graph(self, *, cutoff=0, world=None, frames=None):
        """Expose available topology to Aura's existing analogy machinery.

        Supplied executable frame meanings canonicalize surface symbols and
        port names. They do not establish a meaning for an unknown symbol.
        """
        if type(cutoff) is not int or cutoff < 0:
            raise ValueError("structural analogy requires an available-evidence clock")
        admitted = {v.key for v in (*self.referents, *self.events)
                    if v.available_at <= cutoff and (world is None or v.world == world)}
        relations = []
        for v in self.referents:
            if v.key in admitted:
                relations.append(Relation("type:" + v.type_name, (address(v.key),)))
                relations.append(Relation("world:" + v.world, (address(v.key),)))
        for e in self.events:
            if e.key not in admitted:
                continue
            frame = None if frames is None else frames.get((e.world, e.symbol))
            if frame is not None and (frame.world != e.world or frame.symbol != e.symbol
                    or set(dict(e.ports)) != {r for r, _ in frame.roles}):
                raise ValueError("structural interpretation differs from its symbol ports")
            predicate = e.symbol if frame is None or frame.operation is None else frame.operation
            labels = {} if frame is None or frame.operation is None else {
                role: f"operand:{slot}" for slot, (role, _t) in enumerate(frame.roles)}
            # Reify the event so named port links preserve role/filler direction.
            relations.append(Relation("predicate:" + predicate, (address(e.key),)))
            relations.append(Relation("mode:" + e.modality, (address(e.key),)))
            relations.append(Relation("polarity:" + str(e.polarity), (address(e.key),)))
            relations.append(Relation("world:" + e.world, (address(e.key),)))
            relations.extend(Relation("role:" + labels.get(role, role), (address(e.key), address(key)))
                             for role, key in e.ports if key in admitted)
        relations = tuple(sorted(relations, key=lambda r: (r.predicate, r.args)))
        return Graph(digest([(r.predicate, r.args) for r in relations]), relations)

    def to_dict(self):
        body = {"schema": "aura.meaning_diagram.v1",
            "referents": [{"key": v.key, "type_name": v.type_name, "source": v.source,
                           "available_at": v.available_at, "world": v.world} for v in self.referents],
            "events": [{"key": e.key, "symbol": e.symbol, "ports": e.ports,
                        "source": e.source, "available_at": e.available_at, "world": e.world,
                        "polarity": e.polarity, "modality": e.modality} for e in self.events],
            "temporal": [{"left": c.left, "right": c.right,
                          "lower": None if c.lower is None else str(c.lower),
                          "upper": None if c.upper is None else str(c.upper),
                          "source": c.source, "available_at": c.available_at, "world": c.world}
                         for c in self.temporal]}
        return {**body, "content_sha256": digest(body)}

    @classmethod
    def from_dict(cls, payload):
        body = dict(payload)
        expected = body.pop("content_sha256", None)
        if body.get("schema") != "aura.meaning_diagram.v1" or digest(body) != expected:
            raise ValueError("meaning diagram storage custody differs")
        diagram = cls(tuple(DiagramReferent(**{**v, "key": tuple(v["key"])}) for v in body["referents"]),
            tuple(DiagramEvent(**{**e, "key": tuple(e["key"]),
                "ports": tuple((r, None if k is None else tuple(k)) for r, k in e["ports"])})
                for e in body["events"]), tuple(TemporalConstraint(**c) for c in body["temporal"]))
        if digest(diagram.to_dict()) != digest(payload):
            raise ValueError("meaning diagram storage schema differs")
        return diagram

    def rename(self, identities):
        """Apply an explicit identity bijection; denotation never supplies it."""
        keys = {v.key for v in (*self.referents, *self.events)}
        if set(identities) != keys or len(set(identities.values())) != len(keys):
            raise ValueError("diagram renaming requires a complete identity bijection")
        for key in identities.values():
            source_key(key)
        if self.temporal:
            raise ValueError("temporal endpoint renaming needs its own explicit clock mapping")
        return replace(self, referents=tuple(replace(v, key=identities[v.key]) for v in self.referents),
            events=tuple(replace(e, key=identities[e.key], ports=tuple(
                (r, None if k is None else identities[k]) for r, k in e.ports)) for e in self.events))

    def diff(self, other):
        """Describe a proposed representation change without asserting a world change."""
        if not isinstance(other, MeaningDiagram):
            raise ValueError("diagram difference needs another typed representation")
        left, right = self.to_dict(), other.to_dict()
        changes = {}
        for category in ("referents", "events"):
            before = {tuple(row["key"]): row for row in left[category]}
            after = {tuple(row["key"]): row for row in right[category]}
            changes[category] = [{"key": key, "before": before.get(key), "after": after.get(key)}
                for key in sorted(set(before) | set(after)) if before.get(key) != after.get(key)]
        if left["temporal"] != right["temporal"]:
            changes["temporal"] = {"before": left["temporal"], "after": right["temporal"]}
        body = {"schema": "aura.meaning_diagram_diff.v1", "source": self.custody_sha256,
                "target": other.custody_sha256, "changes": changes, "observed_world_change": False}
        return {**body, "content_sha256": digest(body)}


@dataclass(frozen=True)
class DiagramPresentation:
    """Non-semantic appearance; meaningful proximity must be an explicit edge."""
    diagram: MeaningDiagram
    source: str
    layout: tuple[tuple[tuple[str, str], tuple[float, float]], ...] = ()
    reading_order: tuple[tuple[str, str], ...] = ()
    pronunciation: tuple[str, ...] = ()

    def __post_init__(self):
        import math

        keys = {v.key for v in (*self.diagram.referents, *self.diagram.events)}
        if (not isinstance(self.source, str) or not self.source
                or len({k for k, _ in self.layout}) != len(self.layout)
                or any(k not in keys or len(p) != 2 or not all(map(math.isfinite, p)) for k, p in self.layout)
                or len(set(self.reading_order)) != len(self.reading_order)
                or not set(self.reading_order) <= keys
                or any(not isinstance(p, str) for p in self.pronunciation)):
            raise ValueError("invalid diagram presentation")
        object.__setattr__(self, "layout", tuple((k, tuple(p)) for k, p in self.layout))
        object.__setattr__(self, "reading_order", tuple(self.reading_order))
        object.__setattr__(self, "pronunciation", tuple(self.pronunciation))

    def to_dict(self):
        body = {"schema": "aura.diagram_presentation.v1", "diagram": self.diagram.to_dict(),
                "source": self.source, "layout": self.layout, "reading_order": self.reading_order,
                "pronunciation": self.pronunciation}
        return {**body, "content_sha256": digest(body)}

    @classmethod
    def from_dict(cls, payload):
        body = dict(payload)
        expected = body.pop("content_sha256", None)
        if body.get("schema") != "aura.diagram_presentation.v1" or digest(body) != expected:
            raise ValueError("diagram presentation storage custody differs")
        return cls(MeaningDiagram.from_dict(body["diagram"]), body["source"],
                   tuple((tuple(k), tuple(p)) for k, p in body["layout"]),
                   tuple(tuple(k) for k in body["reading_order"]), tuple(body["pronunciation"]))


@dataclass(frozen=True)
class MeaningFrame:
    symbol: str
    roles: tuple[tuple[str, str], ...]
    output_type: str
    operation: str | None = None
    world: str = "actual"

    def __post_init__(self):
        roles = tuple(self.roles)
        if (any(not isinstance(v, str) or not v for v in (self.symbol, self.output_type, self.world))
                or not roles or len(roles) > 32 or len({r for r, _ in roles}) != len(roles)
                or any(not isinstance(v, str) or not v for row in roles for v in row)):
            raise ValueError("invalid declared meaning frame")
        if self.operation is not None:
            signature = semantic_primitive_type_signature(self.operation)
            if signature != (tuple(t for _, t in roles), self.output_type):
                raise ValueError("meaning frame differs from the typed floor operation")
        object.__setattr__(self, "roles", roles)


def diagram_binding_problem(diagram, frames, *, cutoff, world="actual"):
    """Translate graph ports into the existing contextual role-binding problem."""
    temporal = assess_temporal_constraints(diagram.temporal, cutoff=cutoff, world=world)
    if not temporal.consistent:
        raise ValueError("meaning diagram has inconsistent temporal premises")
    events = [e for e in diagram.events if e.world == world and e.available_at <= cutoff]
    refs = [v for v in diagram.referents if v.world == world and v.available_at <= cutoff]
    types = {v.type_name: None for v in refs}
    records = [ContextReferent(*v.key, v.type_name, v.source, scope=(world,),
                              available_from=v.available_at) for v in refs]
    roles, addresses = [], {}
    for e in events:
        frame = frames.get((world, e.symbol))
        if frame is None or frame.world != world or frame.symbol != e.symbol:
            raise ValueError("meaning event has no declared interpretation in this world")
        if set(dict(e.ports)) != {r for r, _ in frame.roles}:
            raise ValueError("meaning event ports differ from its frame")
        types[frame.output_type] = None
        records.append(ContextReferent(*e.key, frame.output_type, e.source,
            scope=(world,), available_from=e.available_at))
        for slot, (name, type_name) in enumerate(frame.roles):
            types[type_name] = None
            identity = digest((e.key, name))
            filler = dict(e.ports)[name]
            roles.append(BindingRole(identity,
                f"{frame.operation}:operand:{slot}" if frame.operation else name,
                type_name, scope=(world,), tick=cutoff,
                referents=None if filler is None else (filler,)))
            addresses[identity] = (e.key, name)
    if not roles:
        raise ValueError("meaning diagram has no currently available event")
    edges = frozenset((k, e.key) for e in events for _, k in e.ports
                      if k is not None and k in {v.key for v in records})
    return BindingContext(tuple(records), types, edges), tuple(roles), addresses, temporal


def apply_diagram_bindings(diagram, binding, addresses):
    if not binding.executable or set(dict(binding.bindings)) != set(addresses):
        raise ValueError("meaning diagram requires a complete measured binding")
    return apply_diagram_assignment(diagram, binding.bindings, addresses)


def apply_diagram_assignment(diagram, bindings, addresses):
    """Construct a provisional graph; only the binder can admit its assignment."""
    bindings = tuple(bindings)
    if len(bindings) != len(addresses) or set(dict(bindings)) != set(addresses):
        raise ValueError("provisional diagram assignment must cover every role")
    updates = {addresses[identity]: key for identity, key in bindings}
    return replace(diagram, events=tuple(replace(e, ports=tuple(
        (r, updates.get((e.key, r), key)) for r, key in e.ports)) for e in diagram.events))


def compile_meaning_diagram(diagram, frames, inputs, *, output, cutoff, world="actual"):
    """Lower a complete positive asserted DAG to the existing typed SSA floor."""
    context, _, _, temporal = diagram_binding_problem(diagram, frames, cutoff=cutoff, world=world)
    keys = {v.key for v in context.referents}
    by_key = {e.key: e for e in diagram.events if e.world == world and e.available_at <= cutoff}
    if output not in by_key:
        raise ValueError("diagram output is not an available event")
    graph = nx.DiGraph()
    graph.add_nodes_from(by_key)
    for e in by_key.values():
        if not e.polarity or e.modality != "asserted":
            raise ValueError("negated or modal meaning has no asserted execution authority")
        frame = frames[world, e.symbol]
        if frame.operation is None:
            raise ValueError("symbol interpretation has no executable floor denotation")
        for _, k in e.ports:
            if k is None or k not in keys:
                raise ValueError("diagram has an unbound or unavailable role")
            if k in by_key:
                graph.add_edge(k, e.key)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("cyclic meaning requires a declared fixed-point semantics")
    ancestors = nx.ancestors(graph, output) | {output}
    if ancestors != set(by_key):
        raise ValueError("diagram contains events outside the requested causal computation")
    supplied = dict(inputs)
    expected = {v.key for v in diagram.referents if v.world == world and v.available_at <= cutoff}
    if set(supplied) != expected:
        raise ValueError("diagram inputs differ from the available source identities")
    input_keys = tuple(sorted(supplied))
    values = tuple(normalize_semantic_value(supplied[k]) for k in input_keys)
    types = {v.key: v.type_name for v in context.referents}
    for k, value in zip(input_keys, values, strict=True):
        if types[k] != ("integer" if type(value) is int else "integer_sequence"):
            raise ValueError("diagram input value differs from its declared type")
    registers = {k: i for i, k in enumerate(input_keys)}
    instructions = []
    for k in nx.lexicographical_topological_sort(graph, key=address):
        e, frame = by_key[k], frames[world, by_key[k].symbol]
        ports = dict(e.ports)
        if any(types[ports[r]] != t for r, t in frame.roles):
            raise ValueError("diagram role filler violates its declared type")
        instructions.append(Instruction(frame.operation, tuple(registers[ports[r]] for r, _ in frame.roles)))
        registers[k] = len(input_keys) + len(instructions) - 1
    program = Program(len(input_keys), tuple(instructions))
    interpretation_sha256 = digest({"diagram": diagram.custody_sha256,
        "frames": sorted((f.world, f.symbol, f.roles, f.output_type, f.operation)
                         for f in frames.values())})
    compiled = compile_source_independent_program_to_floor(program, values,
        provenance_receipt_sha256=interpretation_sha256)
    return compiled, temporal


def execute_meaning_diagram(diagram, frames, inputs, *, output, cutoff, world="actual", fuel=100_000):
    compiled, _ = compile_meaning_diagram(diagram, frames, inputs,
        output=output, cutoff=cutoff, world=world)
    return execute_semantic_floor_program(compiled, fuel=fuel)


@dataclass(frozen=True)
class DiagramObservations:
    """Actual acquired states, supplied by the perception/model-lane owner."""
    operations: object
    mentions: object
    candidates: object

    def __post_init__(self):
        for name in ("operations", "mentions", "candidates"):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def resolve_meaning_diagram(engine, diagram, frames, observations, *, cutoff,
                            world="actual", baseline_costs=None, diagram_accepts=None, **options):
    from core.learning.semantic_grounded_binding_engine import GroundedBindingEvidence

    context, roles, addresses, temporal = diagram_binding_problem(diagram, frames, cutoff=cutoff, world=world)
    if (set(observations.operations) != {e for e, _ in addresses.values()}
            or set(observations.mentions) != set(addresses.values())
            or set(observations.candidates) != {v.key for v in context.referents}):
        raise ValueError("diagram observations are not aligned to admitted role/source identities")
    evidence = GroundedBindingEvidence(diagram.custody_sha256, context, roles,
        {r.identity: observations.operations[addresses[r.identity][0]] for r in roles},
        {r.identity: observations.mentions[addresses[r.identity]] for r in roles},
        dict(observations.candidates))
    if diagram_accepts is not None:
        if not callable(diagram_accepts) or "assignment_filter" in options:
            raise ValueError("diagram admission needs one declared whole-graph predicate")
        options["assignment_filter"] = lambda bindings: diagram_accepts(
            apply_diagram_assignment(diagram, bindings, addresses))
    binding = engine.resolve(evidence, baseline_costs=baseline_costs, **options)
    resolved = apply_diagram_bindings(diagram, binding, addresses) if binding.executable else None
    return resolved, binding, {"schema": "aura.diagram_resolution.v1",
        "source_diagram": diagram.custody_sha256,
        "resolved_diagram": resolved.to_dict() if resolved is not None else None,
        "binding": engine.binding_receipt(evidence, binding, baseline_costs=baseline_costs),
        "temporal_evidence": temporal.evidence, "excluded_temporal_evidence": temporal.excluded,
        "semantic_truth_certified": False, "serving_authority": False}


@invariant("learning.diagram_layout_does_not_swap_semantic_roles", scope="learning",
           owner="core/learning/semantic_semasiographic.py", observational=False)
def diagram_role_contract():
    refs = (DiagramReferent(("source", "a"), "integer", "observed:a"),
            DiagramReferent(("source", "b"), "integer", "observed:b"))
    event = DiagramEvent(("source", "e"), "glyph", (("left", refs[0].key), ("right", refs[1].key)), "seen")
    diagram = MeaningDiagram(refs, (event,))
    reversed_storage = MeaningDiagram(tuple(reversed(refs)), (replace(event, ports=tuple(reversed(event.ports))),))
    swapped = replace(diagram, events=(replace(event, ports=(("left", refs[1].key), ("right", refs[0].key))),))
    assert diagram.meaning_sha256 == reversed_storage.meaning_sha256
    assert diagram.meaning_sha256 != swapped.meaning_sha256
    return ()


def diagram_event_tensor(diagram, event_key, filler_states):
    """Pack one complete relational event using the existing exact TPR store."""
    from core.learning.semantic_binding_tensor import BindingTensor

    event = next((e for e in diagram.events if e.key == event_key), None)
    if event is None or any(k is None or k not in filler_states for _, k in event.ports):
        raise ValueError("event tensor requires actually observed states for every bound role")
    return BindingTensor.pack({r: filler_states[k] for r, k in event.ports},
                             source_keys=dict(event.ports))
