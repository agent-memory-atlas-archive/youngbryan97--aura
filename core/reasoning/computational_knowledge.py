"""Calculate scoped consequences without upgrading assumptions into observations."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction

from core.engineering.units import DIMENSIONLESS, Dimension, DimensionError, Q, dimension_of
from core.verify.invariants import invariant
from typing import Any


def _sha(value: dict[str, Any]) -> Any:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  allow_nan=False).encode()).hexdigest()


def _rational(value: Any) -> Any:
    if type(value) not in {int, float, str, Fraction} or isinstance(value, bool):
        raise ValueError("quantity bounds must be finite rational numbers")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("quantity bounds must be finite rational numbers")
    if type(value) is str and len(value) > 2048:
        raise ValueError("quantity arithmetic exceeds the declared size bound")
    result = Fraction(str(value)) if type(value) is float else Fraction(value)
    if max(result.numerator.bit_length(), result.denominator.bit_length()) > 4096:
        raise ValueError("quantity arithmetic exceeds the declared size bound")
    return result


@dataclass(frozen=True, slots=True)
class QuantityBounds:
    """An enclosing rational interval in SI units, with the shared SI dimensions."""

    lower: Fraction
    upper: Fraction
    dimension: Dimension = DIMENSIONLESS

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", _rational(self.lower))
        object.__setattr__(self, "upper", _rational(self.upper))
        if self.lower > self.upper or not isinstance(self.dimension, Dimension):
            raise ValueError("quantity bounds or dimensions are invalid")

    @classmethod
    def of(cls, lower: int, upper: int=None, unit: str='count') -> Any:
        zero, one = Q(0, unit), Q(1, unit)
        if zero.value != 0 or one.value <= 0:
            raise ValueError("use absolute SI values for offset units")
        scale = _rational(one.value)
        return cls(_rational(lower) * scale,
                   _rational(lower if upper is None else upper) * scale, one.dimension)

    def to_dict(self) -> dict[str, Any]:
        return {"lower": str(self.lower), "upper": str(self.upper),
                "si_dimension": [str(value) for value in self.dimension.exponents]}

    def __add__(self, other: Any) -> Any:
        if self.dimension != other.dimension:
            raise DimensionError("computed sum has incompatible dimensions")
        return QuantityBounds(self.lower + other.lower, self.upper + other.upper, self.dimension)

    def __neg__(self) -> Any:
        return QuantityBounds(-self.upper, -self.lower, self.dimension)

    def __sub__(self, other: Any) -> Any:
        return self + -other

    def __mul__(self, other: Any) -> Any:
        products = [left * right for left in (self.lower, self.upper)
                    for right in (other.lower, other.upper)]
        return QuantityBounds(min(products), max(products), self.dimension * other.dimension)

    def __truediv__(self, other: Any) -> Any:
        if other.lower <= 0 <= other.upper:
            raise ValueError("division interval includes zero; consequence is unproved")
        reciprocal = QuantityBounds(1 / other.upper, 1 / other.lower,
                                    DIMENSIONLESS / other.dimension)
        return self * reciprocal

    def __pow__(self, power: Any) -> Any:
        if type(power) is not int or not -4 <= power <= 4:
            raise ValueError("computed power exceeds the declared expression grammar")
        if power < 0:
            return QuantityBounds.of(1) / self ** -power
        values = [self.lower ** power, self.upper ** power]
        if power > 0 and power % 2 == 0 and self.lower <= 0 <= self.upper:
            values.append(Fraction(0))
        return QuantityBounds(min(values), max(values), self.dimension ** power)


@dataclass(frozen=True, slots=True)
class ScopedPremise:
    """A caller-identified observation, given, estimate, or hypothetical premise."""

    identity: str
    scope: str
    value: QuantityBounds | bool
    origin: str
    ref: str
    kind: str = "given"
    proposition: str = ""
    valid_from: float | None = None
    valid_until: float | None = None
    dependencies: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (any(not isinstance(value, str) or not value.strip()
                for value in (self.identity, self.scope, self.origin, self.ref))
                or not isinstance(self.value, QuantityBounds) and type(self.value) is not bool
                or self.kind not in {"given", "measurement", "estimate", "assumption"}
                or not isinstance(self.proposition, str)
                or type(self.value) is bool and not self.proposition
                or not isinstance(self.dependencies, tuple)
                or len(set(self.dependencies)) != len(self.dependencies)
                or any(not isinstance(item, str) or not item for item in self.dependencies)):
            raise ValueError("computed knowledge premise lacks scope or provenance")
        for timestamp in (self.valid_from, self.valid_until):
            if timestamp is not None and (type(timestamp) not in {int, float}
                                         or not math.isfinite(timestamp)):
                raise ValueError("premise validity requires finite timestamps")
        if (self.valid_from is not None and self.valid_until is not None
                and self.valid_from > self.valid_until):
            raise ValueError("premise validity interval is reversed")

    def to_dict(self) -> dict[str, Any]:
        return {"identity": self.identity, "scope": self.scope, "origin": self.origin,
                "ref": self.ref, "kind": self.kind, "proposition": self.proposition,
                "value": self.value.to_dict() if isinstance(self.value, QuantityBounds) else self.value,
                "valid_from": self.valid_from, "valid_until": self.valid_until,
                "dependencies": self.dependencies}


@dataclass(frozen=True, slots=True)
class KnowledgeContext:
    scope: str
    now: float
    premises: tuple[ScopedPremise, ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.scope, str) or not self.scope
                or type(self.now) not in {int, float} or not math.isfinite(self.now)
                or not isinstance(self.premises, tuple) or len(self.premises) > 256
                or any(not isinstance(item, ScopedPremise) for item in self.premises)
                or len({item.identity for item in self.premises}) != len(self.premises)):
            raise ValueError("computed knowledge context is invalid or repeats a premise")

    def premise(self, identity: Any) -> Any:
        item = next((item for item in self.premises if item.identity == identity), None)
        if (item is None or item.scope != self.scope
                or item.valid_from is not None and self.now < item.valid_from
                or item.valid_until is not None and self.now > item.valid_until):
            raise ValueError("computed knowledge premise is absent, stale, or in another scope")
        return item

    def to_dict(self) -> dict[str, Any]:
        return {"scope": self.scope, "now": self.now,
                "premises": [item.to_dict() for item in self.premises]}


def _expression(text: Any, values: dict[str, Any]) -> Any:
    tree = ast.parse(text, mode="eval")
    if len(tuple(ast.walk(tree))) > 128:
        raise ValueError("computed expression exceeds its node bound")

    def walk(node: Any) -> Any:
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            return QuantityBounds.of(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                return walk(node.left) ** node.right.value
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("computed expression contains an unsupported operation")
    return walk(tree.body)


def _expression_dimension(text: Any, units: dict[str, Any]) -> Any:
    tree = ast.parse(text, mode="eval")
    if len(tuple(ast.walk(tree))) > 128:
        raise ValueError("computed expression exceeds its node bound")

    def walk(node: Any) -> Any:
        if isinstance(node, ast.Name):
            if node.id not in units:
                raise ValueError("computed expression names an undeclared input")
            return dimension_of(units[node.id])
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            _rational(node.value)
            return DIMENSIONLESS
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return walk(node.operand)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant):
                power = node.right.value
                QuantityBounds.of(1) ** power
                return walk(node.left) ** power
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, (ast.Add, ast.Sub)):
                if left != right:
                    raise DimensionError("computed expression combines incompatible dimensions")
                return left
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
        raise ValueError("computed expression contains an unsupported operation")
    return walk(tree.body)


def premise_closure(context: KnowledgeContext, identities: tuple[Any, ...]) -> Any:
    """A derived value keeps the freshness and uncertainty of all its ancestors."""
    checked, active = {}, set()

    def visit(identity: Any) -> None:
        if identity in active:
            raise ValueError("computed knowledge premise dependencies are cyclic")
        if identity in checked:
            return
        item = context.premise(identity)
        active.add(identity)
        for ancestor in item.dependencies:
            visit(ancestor)
        active.remove(identity)
        checked[identity] = item
    for identity in identities:
        visit(identity)
    return checked


@dataclass(frozen=True, slots=True)
class ComputationModel:
    """A versioned formula; model validity is an explicit premise, never a default."""

    identity: str
    field: str
    inputs: tuple[tuple[str, str], ...]
    output_unit: str
    expression: str
    assumptions: tuple[str, ...]
    reference: str
    nonnegative: tuple[str, ...] = ()
    positive: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (any(not isinstance(value, str) or not value.strip()
                for value in (self.identity, self.field, self.reference, self.output_unit, self.expression))
                or not isinstance(self.inputs, tuple) or len(self.inputs) > 64
                or any(not isinstance(pair, tuple) or len(pair) != 2
                    or any(not isinstance(value, str) or not value for value in pair)
                    for pair in self.inputs)
                or any(not isinstance(items, tuple) or any(not isinstance(value, str) or not value
                    for value in items) for items in (self.assumptions, self.nonnegative, self.positive))):
            raise ValueError("computed model declaration must be immutable named data")
        names = tuple(name for name, _unit in self.inputs)
        if (not self.identity or not self.field or not self.reference or not self.inputs
                or len(names) != len(set(names)) or len(self.expression) > 1024
                or len(set(self.assumptions)) != len(self.assumptions)
                or not set((*self.nonnegative, *self.positive)) <= set(names)):
            raise ValueError("computed model declaration is invalid")
        # Declaration checks must not assume arbitrary numeric inputs are valid.
        if _expression_dimension(self.expression, dict(self.inputs)) != dimension_of(self.output_unit):
            raise DimensionError("computed model output dimension differs")

    @property
    def identity_sha256(self) -> Any:
        return _sha({name: getattr(self, name) for name in self.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class ComputationRequest:
    identity: str
    model: ComputationModel
    bindings: tuple[tuple[str, str], ...]
    assumption_bindings: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.identity, str) or not self.identity
                or not isinstance(self.model, ComputationModel)
                or any(not isinstance(rows, tuple) or any(not isinstance(row, tuple)
                    or len(row) != 2 or any(not isinstance(value, str) or not value for value in row)
                    for row in rows) for rows in (self.bindings, self.assumption_bindings))):
            raise ValueError("computation request lacks a model or named premise bindings")


@dataclass(frozen=True, slots=True)
class ComputedKnowledge:
    request: ComputationRequest
    context: KnowledgeContext
    result: QuantityBounds
    premise_ids: tuple[str, ...]
    hard_constraint: bool

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": "aura.computed_knowledge.v1", "identity": self.request.identity,
                "scope": self.context.scope, "context_sha256": _sha(self.context.to_dict()),
                "model_sha256": self.request.model.identity_sha256,
                "model": self.request.model.identity, "reference": self.request.model.reference,
                "bindings": self.request.bindings, "assumption_bindings": self.request.assumption_bindings,
                "premise_ids": self.premise_ids, "result": self.result.to_dict(),
                "hard_constraint": self.hard_constraint,
                "claim": "enclosing_consequence_conditional_on_scoped_premises_and_model",
                "source_interpretation_proven": False, "general_transfer_proven": False}
        return {**body, "receipt_sha256": _sha(body)}


def compute_knowledge(request: ComputationRequest, context: KnowledgeContext) -> ComputedKnowledge:
    """Execute bounded arithmetic and retain every input and applicability premise."""
    if not isinstance(request, ComputationRequest) or not request.identity:
        raise ValueError("computed knowledge requires an identified request")
    model = request.model
    bindings, guards = dict(request.bindings), dict(request.assumption_bindings)
    if (len(bindings) != len(request.bindings) or len(guards) != len(request.assumption_bindings)
            or set(bindings) != {name for name, _unit in model.inputs}
            or set(guards) != set(model.assumptions)):
        raise ValueError("computed knowledge bindings or model assumptions are incomplete")
    premises = premise_closure(context, (*bindings.values(), *guards.values()))
    used = tuple(sorted(premises))
    for proposition, identity in guards.items():
        item = premises[identity]
        if item.proposition != proposition or item.value is not True:
            raise ValueError("computed model applicability is contradicted or unestablished")
        for peer in context.premises:
            if (peer.scope == context.scope and peer.proposition == proposition and peer.value is False
                    and peer.kind in {"given", "measurement"}
                    and (peer.valid_from is None or context.now >= peer.valid_from)
                    and (peer.valid_until is None or context.now <= peer.valid_until)):
                raise ValueError("computed model applicability has contradictory evidence")
    values = {}
    for name, unit in model.inputs:
        value = premises[bindings[name]].value
        if not isinstance(value, QuantityBounds) or value.dimension != dimension_of(unit):
            raise DimensionError("computed input differs from the required quantity dimension")
        if (name in model.nonnegative and value.lower < 0
                or name in model.positive and value.lower <= 0):
            raise ValueError("computed inputs extend outside the model's declared domain")
        values[name] = value
    result = _expression(model.expression, values)
    if result.dimension != dimension_of(model.output_unit):
        raise DimensionError("computed consequence lost the model's dimensions")
    return ComputedKnowledge(request, context, result, used,
                             all(item.kind in {"given", "measurement"} for item in premises.values()))


def verify_computed_knowledge(knowledge: ComputedKnowledge, *, scope: str, now: float) -> Any:
    """Recompute the consequence and recheck scope and time before selection."""
    if not isinstance(knowledge, ComputedKnowledge) or knowledge.context.scope != scope:
        raise ValueError("computed consequence belongs to another problem")
    refreshed = KnowledgeContext(scope, now, knowledge.context.premises)
    replay = compute_knowledge(knowledge.request, refreshed)
    if (replay.result != knowledge.result or replay.premise_ids != knowledge.premise_ids
            or replay.hard_constraint != knowledge.hard_constraint):
        raise ValueError("computed consequence differs from arithmetic replay")
    return replay


async def compute_knowledge_batch(requests: tuple[Any, ...], context: Any) -> Any:
    """Move independent calculation cells off the live loop; preserve request order."""
    from core.runtime.executors import off_the_loop
    requests = tuple(requests)
    if (not requests or len(requests) > 128
            or len({request.identity for request in requests}) != len(requests)):
        raise ValueError("computed batch is empty, excessive, or repeats a cell")
    return await off_the_loop(lambda: tuple(compute_knowledge(request, context) for request in requests))


@invariant("reasoning.computation_does_not_promote_estimates", scope="reasoning",
           owner="core/reasoning/computational_knowledge.py", observational=False)
def _estimate_is_not_exact() -> Any:
    context = KnowledgeContext("request", 0., (
        ScopedPremise("speed", "request", QuantityBounds.of(9, 11, "m/s"), "sensor", "speed:1", "estimate"),
        ScopedPremise("time", "request", QuantityBounds.of(2, unit="s"), "clock", "time:1"),
    ))
    model = ComputationModel("constant_speed.v1", "motion", (("v", "m/s"), ("t", "s")),
                             "m", "v*t", (), "definition of constant speed")
    measured = compute_knowledge(ComputationRequest("distance", model, (("v", "speed"), ("t", "time"))), context)
    assert measured.result == QuantityBounds.of(18, 22, "m") and not measured.hard_constraint
    return measured.to_dict()
