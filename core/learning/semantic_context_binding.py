"""Resolve supplied role evidence against a typed, source-addressed context.

This layer does not infer a frame from arbitrary English. Learned parsers,
retrievers and program charts supply roles and costs. Identity, scope and
declared relations constrain those costs without replacing their evidence.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.verify.invariants import invariant

SourceKey = tuple[str, str]


@dataclass(frozen=True)
class ContextReferent:
    namespace: str
    identity: str
    type_name: str
    source: str
    scope: tuple[str, ...] = ()
    available_from: int = 0
    available_until: int | None = None
    aliases: tuple[str, ...] = ()
    attributes: Mapping[str, str] = field(default_factory=dict)
    embedding: tuple[float, ...] = ()

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value for value in
                (self.namespace, self.identity, self.type_name, self.source))
                or type(self.available_from) is not int or self.available_from < 0
                or (self.available_until is not None and
                    (type(self.available_until) is not int or self.available_until <= self.available_from))
                or any(not isinstance(value, str) or not value for value in (*self.scope, *self.aliases))
                or any(not isinstance(key, str) or not isinstance(value, str)
                       for key, value in self.attributes.items())):
            raise ValueError("invalid contextual referent")
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "embedding", _vector(self.embedding))

    @property
    def key(self) -> SourceKey:
        return self.namespace, self.identity


def _vector(values):
    values = tuple(float(value) for value in values)
    if values and (not all(map(math.isfinite, values)) or not any(values)):
        raise ValueError("binding vectors must be finite and nonzero")
    return values


@dataclass(frozen=True)
class BindingRole:
    identity: str
    role: str
    type_name: str
    scope: tuple[str, ...] = ()
    tick: int = 0
    referents: tuple[SourceKey, ...] | None = None
    agreement: Mapping[str, str] = field(default_factory=dict)
    embedding: tuple[float, ...] = ()
    unbound_cost: float | None = None

    def __post_init__(self):
        if (any(not isinstance(value, str) or not value for value in
                (self.identity, self.role, self.type_name))
                or type(self.tick) is not int or self.tick < 0
                or any(not isinstance(value, str) or not value for value in self.scope)
                or any(not isinstance(key, str) or not isinstance(value, str)
                       for key, value in self.agreement.items())
                or (self.unbound_cost is not None and not math.isfinite(self.unbound_cost))):
            raise ValueError("invalid binding role")
        object.__setattr__(self, "scope", tuple(self.scope))
        object.__setattr__(self, "agreement", MappingProxyType(dict(self.agreement)))
        object.__setattr__(self, "embedding", _vector(self.embedding))
        if self.referents is not None:
            keys = tuple(tuple(key) for key in self.referents)
            if any(len(key) != 2 or any(not isinstance(part, str) or not part for part in key)
                   for key in keys):
                raise ValueError("invalid role source identity")
            object.__setattr__(self, "referents", keys)


@dataclass(frozen=True)
class BindingContext:
    referents: tuple[ContextReferent, ...]
    type_parents: Mapping[str, str | None]
    edges: frozenset[tuple[SourceKey, SourceKey]] = frozenset()

    def __post_init__(self):
        referents, parents = tuple(self.referents), dict(self.type_parents)
        keys = {item.key for item in referents}
        if (len(keys) != len(referents) or not parents
                or any(not isinstance(key, str) or not key or
                       (parent is not None and parent not in parents) for key, parent in parents.items())
                or any(item.type_name not in parents for item in referents)
                or any(left not in keys or right not in keys for left, right in self.edges)):
            raise ValueError("invalid binding context identity or types")
        for name in parents:
            seen = set()
            while name is not None:
                if name in seen:
                    raise ValueError("binding type hierarchy contains a cycle")
                seen.add(name)
                name = parents[name]
        object.__setattr__(self, "referents", referents)
        object.__setattr__(self, "type_parents", MappingProxyType(parents))
        object.__setattr__(self, "edges", frozenset(self.edges))

    def eligible(self, role: BindingRole, candidate: ContextReferent) -> bool:
        if role.type_name not in self.type_parents:
            return False
        name = candidate.type_name
        while name is not None and name != role.type_name:
            name = self.type_parents[name]
        return (name == role.type_name
                and role.scope[:len(candidate.scope)] == candidate.scope
                and candidate.available_from <= role.tick
                and (candidate.available_until is None or role.tick < candidate.available_until)
                and (role.referents is None or candidate.key in role.referents)
                and all(candidate.attributes.get(key) == value for key, value in role.agreement.items()))

    def alias_candidates(self, alias: str, role: BindingRole) -> tuple[SourceKey, ...]:
        normalized = " ".join(alias.casefold().split())
        return tuple(sorted(candidate.key for candidate in self.referents
                            if normalized and self.eligible(role, candidate)
                            and normalized in {" ".join(value.casefold().split()) for value in candidate.aliases}))

    def reaches(self, left: SourceKey, right: SourceKey) -> bool:
        pending, seen = [left], {left}
        while pending:
            current = pending.pop()
            for source, target in self.edges:
                if source == current:
                    if target == right:
                        return True
                    if target not in seen:
                        seen.add(target)
                        pending.append(target)
        return False


@dataclass(frozen=True)
class BindingRelation:
    left: str
    right: str
    relation: str
    violation_cost: float | None = None

    def __post_init__(self):
        if (not self.left or not self.right or self.left == self.right
                or self.relation not in {"same", "different", "reaches"}
                or (self.violation_cost is not None and
                    (not math.isfinite(self.violation_cost) or self.violation_cost < 0))):
            raise ValueError("invalid binding relation")

    def cost(self, context: BindingContext, left: SourceKey | None, right: SourceKey | None):
        # Unknown fillers do not prove or disprove a relation; required roles
        # cannot choose them, and optional roles remain visibly unbound.
        if left is None or right is None:
            return 0.
        valid = ((left == right) if self.relation == "same" else
                 (left != right) if self.relation == "different" else context.reaches(left, right))
        return 0. if valid else self.violation_cost


def candidate_cost(context, role, candidate, *, observed_cost=None):
    if not context.eligible(role, candidate):
        return None
    if observed_cost is not None and not math.isfinite(observed_cost):
        raise ValueError("nonfinite binding evidence")
    cost = observed_cost
    if role.embedding and candidate.embedding:
        if len(role.embedding) != len(candidate.embedding):
            raise ValueError("binding evidence widths differ")
        # Normalize after rescaling to avoid overflow for finite large vectors.
        left, right = np.asarray(role.embedding), np.asarray(candidate.embedding)
        left, right = left / np.max(np.abs(left)), right / np.max(np.abs(right))
        distance = 1. - float(np.dot(left / np.linalg.norm(left), right / np.linalg.norm(right)))
        cost = (0. if cost is None else cost) + distance
    if cost is None and role.referents is not None and candidate.key in role.referents:
        cost = 0.
    return cost


@dataclass(frozen=True)
class GroundedBinding:
    bindings: tuple[tuple[str, SourceKey | None], ...]
    energy: float | None
    alternatives: tuple[tuple[str, SourceKey | None], ...]
    margin: float | None
    status: str
    candidate_counts: tuple[int, ...]

    @property
    def executable(self):
        return self.status == "bound"


def bind_context_roles(context: BindingContext, roles: Sequence[BindingRole], *,
                       costs: Mapping[tuple[str, SourceKey], float] | None = None,
                       fallback_costs: Mapping[tuple[str, SourceKey], float] | None = None,
                       parser_confidence: float = 1., parser_threshold: float = 0.,
                       relations: Sequence[BindingRelation] = (), minimum_margin: float = 0.,
                       time_limit_s: float = 5., node_limit: int = 10000,
                       assignment_filter=None, assignment_limit: int = 256) -> GroundedBinding:
    """Optimize all admitted sources; measure the best distinct assignment too.

    Confidence chooses a supplied evidence source, never a type/scope bypass.
    The margin is an energy gap, not a calibrated probability of correctness.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    roles, relations = tuple(roles), tuple(relations)
    names = {role.identity for role in roles}
    if (not roles or len(names) != len(roles) or not math.isfinite(minimum_margin) or minimum_margin < 0
            or not 0 <= parser_confidence <= 1 or not 0 <= parser_threshold <= 1
            or not math.isfinite(time_limit_s) or time_limit_s <= 0
            or type(node_limit) is not int or node_limit < 1
            or assignment_filter is not None and not callable(assignment_filter)
            or type(assignment_limit) is not int or not 1 <= assignment_limit <= 4096
            or any(relation.left not in names or relation.right not in names for relation in relations)):
        raise ValueError("invalid contextual binding problem")
    deadline = time.monotonic() + time_limit_s
    evidence = (fallback_costs if parser_confidence < parser_threshold else costs) or {}
    choices, groups = [], {}
    for role in roles:
        if time.monotonic() >= deadline:
            raise ArgumentOptimizationIncompleteError("context_binding_preparation_budget_exhausted")
        indices = []
        for candidate in context.referents:
            if time.monotonic() >= deadline:
                raise ArgumentOptimizationIncompleteError("context_binding_preparation_budget_exhausted")
            cost = candidate_cost(context, role, candidate,
                                  observed_cost=evidence.get((role.identity, candidate.key)))
            if cost is not None:
                indices.append(len(choices))
                choices.append((role.identity, candidate.key, cost))
        if role.unbound_cost is not None:
            indices.append(len(choices))
            choices.append((role.identity, None, role.unbound_cost))
        groups[role.identity] = indices
    counts = tuple(sum(choices[index][1] is not None for index in groups[role.identity]) for role in roles)
    if any(not group for group in groups.values()):
        return GroundedBinding((), None, (), None, "unsupported", counts)
    objective = [cost for _name, _key, cost in choices]
    rows = []
    for indices in groups.values():
        rows.append((dict.fromkeys(indices, 1.), 1., 1.))
    for relation in relations:
        for left in groups[relation.left]:
            if time.monotonic() >= deadline:
                raise ArgumentOptimizationIncompleteError("context_binding_preparation_budget_exhausted")
            for right in groups[relation.right]:
                if time.monotonic() >= deadline:
                    raise ArgumentOptimizationIncompleteError("context_binding_preparation_budget_exhausted")
                cost = relation.cost(context, choices[left][1], choices[right][1])
                if cost is None:
                    rows.append(({left: 1., right: 1.}, -np.inf, 1.))
                elif cost:
                    variable = len(objective)
                    objective.append(cost)
                    rows.extend((({variable: 1., left: -1.}, -np.inf, 0.),
                                 ({variable: 1., right: -1.}, -np.inf, 0.),
                                 ({variable: 1., left: -1., right: -1.}, -1., np.inf)))

    def solve(excluded=()):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ArgumentOptimizationIncompleteError("context_binding_budget_exhausted")
        constraints = rows + [(dict.fromkeys(indices, 1.), -np.inf, len(roles) - 1)
                              for indices in excluded]
        ri, ci, values = [], [], []
        for index, (row, _lo, _hi) in enumerate(constraints):
            for column, value in row.items():
                ri.append(index)
                ci.append(column)
                values.append(value)
        matrix = coo_matrix((values, (ri, ci)), shape=(len(constraints), len(objective))).tocsc()
        lows, highs = np.array([row[1] for row in constraints]), np.array([row[2] for row in constraints])
        result = milp(objective, integrality=np.ones(len(objective)), bounds=Bounds(0., 1.),
                      constraints=LinearConstraint(matrix, lows, highs),
                      options={"time_limit": remaining, "node_limit": node_limit, "mip_rel_gap": 0.})
        if result.status == 2:
            return None
        if result.status != 0 or result.x is None:
            raise ArgumentOptimizationIncompleteError(f"context_binding_solver_status:{result.status}")
        point = np.asarray(result.x)
        if (point.shape != (len(objective),) or not np.all(np.isfinite(point))
                or np.any(np.abs(point - np.rint(point)) > 1e-6)
                or np.any(point < -1e-6) or np.any(point > 1 + 1e-6)
                or np.any(matrix @ point < lows - 1e-6) or np.any(matrix @ point > highs + 1e-6)):
            raise ValueError("context binder returned an invalid integer assignment")
        selected = tuple(index for index in range(len(choices)) if point[index] > .5)
        bindings = tuple(sorted((choices[index][0], choices[index][1]) for index in selected))
        return selected, bindings, float(np.dot(objective, np.rint(point)))

    cuts, attempts = [], 0

    def next_admitted():
        nonlocal attempts
        while True:
            result = solve(cuts)
            if result is None or assignment_filter is None:
                return result
            if attempts >= assignment_limit:
                raise ArgumentOptimizationIncompleteError("context_binding_assignment_filter_budget_exhausted")
            attempts += 1
            admitted = assignment_filter(result[1])
            if type(admitted) is not bool:
                raise ValueError("whole-assignment admission must return an explicit bool")
            if admitted:
                return result
            cuts.append(result[0])

    best = next_admitted()
    if best is None:
        return GroundedBinding((), None, (), None, "infeasible", counts)
    cuts.append(best[0])
    alternative = next_admitted()
    margin = None if alternative is None else alternative[2] - best[2]
    tolerance = 1e-7 * (1. + sum(abs(value) for value in objective))
    ambiguous = margin is not None and margin <= minimum_margin + tolerance
    status = "ambiguous" if ambiguous else "partial" if any(key is None for _name, key in best[1]) else "bound"
    return GroundedBinding(best[1], best[2], () if alternative is None else alternative[1], margin, status, counts)


def binding_margin_survives(margin: float, role_energy_bounds: Sequence[float]) -> bool:
    """Sufficient argmin stability when the feasible set and relations stay fixed.

    Each complete assignment can move by sum(bounds). Its gap can shrink by
    twice that amount. This certifies score stability, not intended meaning.
    """
    if not math.isfinite(margin) or any(not math.isfinite(value) or value < 0 for value in role_energy_bounds):
        raise ValueError("invalid binding perturbation bound")
    return margin > 2. * math.fsum(role_energy_bounds)


@dataclass(frozen=True)
class GroundedChartResolution:
    assignment: tuple | None
    bindings: tuple[tuple[str, SourceKey], ...]
    margin: float | None
    status: str


def solve_grounded_argument_chart(chart, context: BindingContext, roles, register_keys, *,
                                  relations=(), minimum_margin=0., time_limit_s=5., costs=None):
    """Add context evidence to the existing connected, acyclic SSA solve.

    No source string is regenerated and no local winner truncates the chart.
    Definition consistency and mention exclusivity remain in the same solve.
    """
    from core.learning.semantic_argument_optimization import optimize_argument_chart

    roles = tuple(tuple(node) for node in roles)
    costs = {} if costs is None else costs
    if (len(roles) != len(chart.options) or any(len(node) != len(options)
            for node, options in zip(roles, chart.options, strict=True))
            or len({role.identity for node in roles for role in node}) != sum(map(len, roles))
            or not math.isfinite(minimum_margin) or minimum_margin < 0
            or not math.isfinite(time_limit_s) or time_limit_s <= 0):
        raise ValueError("context roles differ from argument chart")
    deadline = time.monotonic() + time_limit_s
    candidates = {candidate.key: candidate for candidate in context.referents}
    if (any(type(register) is not int or not 0 <= register < chart.n_inputs + len(chart.options)
            or key not in candidates for register, key in register_keys.items())
            or len(set(register_keys.values())) != len(register_keys)):
        raise ValueError("register source identities must be explicit and distinct")
    options, definitions, addresses = [], [], {}
    for node_index, (node, role_node) in enumerate(zip(chart.options, roles, strict=True)):
        if time.monotonic() >= deadline:
            raise ArgumentOptimizationIncompleteError("grounded_chart_preparation_budget_exhausted")
        row, labels = [], []
        for position, (slot, role) in enumerate(zip(node, role_node, strict=True)):
            pool, names = [], []
            addresses[role.identity] = []
            for index, (score, register, span) in enumerate(slot):
                if time.monotonic() >= deadline:
                    raise ArgumentOptimizationIncompleteError("grounded_chart_preparation_budget_exhausted")
                if register not in register_keys:
                    raise ValueError("argument candidate has no contextual source identity")
                key = register_keys[register]
                cost = candidate_cost(context, role, candidates[key],
                                      observed_cost=costs.get((role.identity, key), 0.))
                if cost is None:
                    continue
                addresses[role.identity].append(((node_index, position, len(pool)), key))
                pool.append((score - cost, register, span))
                if chart.definition_options is not None:
                    names.append(chart.definition_options[node_index][position][index])
            row.append(tuple(pool))
            labels.append(tuple(names))
        options.append(tuple(row))
        definitions.append(tuple(labels))
    pair_costs = {}
    for relation in relations:
        if relation.left not in addresses or relation.right not in addresses:
            raise ValueError("relation has no chart role")
        for left, left_key in addresses[relation.left]:
            if time.monotonic() >= deadline:
                raise ArgumentOptimizationIncompleteError("grounded_chart_preparation_budget_exhausted")
            for right, right_key in addresses[relation.right]:
                if time.monotonic() >= deadline:
                    raise ArgumentOptimizationIncompleteError("grounded_chart_preparation_budget_exhausted")
                cost = relation.cost(context, left_key, right_key)
                if cost is None or cost:
                    pair = tuple(sorted((left, right)))
                    prior = pair_costs.get(pair, 0.)
                    pair_costs[pair] = None if prior is None or cost is None else prior - cost

    def solve(excluded=()):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ArgumentOptimizationIncompleteError("grounded_chart_budget_exhausted")
        return optimize_argument_chart(options, n_inputs=chart.n_inputs, contract=chart.contract,
            definition_options=tuple(definitions) if chart.definition_options is not None else None,
            definition_scores=chart.definition_scores, prune_dominated=chart.prune_dominated,
            option_pair_factors=tuple((left, right, cost) for (left, right), cost in pair_costs.items()),
            excluded_graphs=excluded, time_limit_s=remaining)

    best = solve()
    if best is None:
        return GroundedChartResolution(None, (), None, "infeasible")
    alternative = solve((best[1],))
    margin = None if alternative is None else best[0] - alternative[0]
    tolerance = 1e-7 * (1. + sum(abs(option[0]) for node in options for slot in node for option in slot)
                        + sum(abs(cost) for cost in pair_costs.values() if cost is not None))
    status = "ambiguous" if margin is not None and margin <= minimum_margin + tolerance else "bound"
    bindings = tuple(sorted((role.identity, register_keys[register])
        for node, node_roles in zip(best[1], roles, strict=True)
        for register, role in zip(node, node_roles, strict=True)))
    assignment = (best[0] - chart.choice_log_normalizer, *best[1:])
    return GroundedChartResolution(assignment, bindings, margin, status)


def triadic_context_costs(context, roles, heads, operations, mentions, definitions):
    """Carry a real fitted triadic scorer into contextual/SSA optimization.

    Keys are opaque source identities. Missing vectors remain unsupported,
    and contextual eligibility is checked before scoring a candidate.
    """
    costs = {}
    for role in roles:
        if role.identity not in heads or role.identity not in operations or role.identity not in mentions:
            continue
        for candidate in context.referents:
            if context.eligible(role, candidate) and candidate.key in definitions:
                score = heads[role.identity].score(np.asarray(operations[role.identity]),
                    np.asarray(mentions[role.identity]), np.asarray(definitions[candidate.key]))
                if not math.isfinite(score):
                    raise ValueError("nonfinite fitted contextual evidence")
                costs[role.identity, candidate.key] = -score
    return costs


@invariant("learning.context_binding_preserves_source_identity", scope="learning", owner=__name__)
def source_identity_is_not_denotation():
    context = BindingContext((ContextReferent("turn1", "result", "Number", "node:0"),
                              ContextReferent("turn2", "literal", "Number", "token:0")), {"Number": None})
    role = BindingRole("operand", "minuend", "Number", referents=(("turn1", "result"),))
    assert tuple(candidate.key for candidate in context.referents if context.eligible(role, candidate)) == (
        ("turn1", "result"),)
    return ()
