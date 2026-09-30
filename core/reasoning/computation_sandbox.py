"""Compose reviewed calculation cells and retain reusable recipes with open gaps."""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace

from core.engineering.units import dimension_of
from core.reasoning.computational_knowledge import (
    ComputationModel,
    ComputationRequest,
    ComputedKnowledge,
    KnowledgeContext,
    QuantityBounds,
    ScopedPremise,
    _expression,
    _rational,
    _sha,
    compute_knowledge,
)
from typing import Any


@dataclass(frozen=True, slots=True)
class Formulation:
    """A mathematical dataflow graph with a semantic definition, not an answer cache."""

    identity: str
    definition: str
    cells: tuple[ComputationRequest, ...]
    outputs: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(cell.identity for cell in self.cells)
        if (not self.identity or not self.definition.strip() or len(self.definition) > 8192
                or not self.cells or len(self.cells) > 128 or len(names) != len(set(names))
                or not self.outputs or not set(self.outputs) <= set(names)):
            raise ValueError("formulation needs unique cells, outputs, and a bounded definition")

    def to_dict(self) -> dict[str, Any]:
        body = {"schema": "aura.computation_formulation.v1", "identity": self.identity,
                "definition": self.definition, "outputs": self.outputs,
                "cells": [{"identity": cell.identity, "model": cell.model.identity,
                           "model_sha256": cell.model.identity_sha256,
                           "bindings": cell.bindings, "assumption_bindings": cell.assumption_bindings}
                          for cell in self.cells],
                "semantic_mapping_authority": "caller_supplied_unproved",
                "retained_answers_available": False, "general_transfer_proven": False}
        return {**body, "receipt_sha256": _sha(body)}

    @classmethod
    def from_dict(cls, payload: Any, catalog: Any) -> Any:
        if payload.get("schema") != "aura.computation_formulation.v1":
            raise ValueError("retained formulation schema differs")
        cells = []
        for row in payload["cells"]:
            model = catalog[row["model"]]
            if model.identity_sha256 != row["model_sha256"]:
                raise ValueError("retained formulation model changed")
            cells.append(ComputationRequest(row["identity"], model,
                tuple(tuple(pair) for pair in row["bindings"]),
                tuple(tuple(pair) for pair in row["assumption_bindings"])))
        recipe = cls(payload["identity"], payload["definition"], tuple(cells), tuple(payload["outputs"]))
        if _sha(recipe.to_dict()) != _sha(payload):
            raise ValueError("retained formulation content differs")
        return recipe

    async def retain(self, gateway: Any) -> Any:
        from core.runtime.gateways import StateMutationRequest
        payload = self.to_dict()
        return await gateway.mutate(StateMutationRequest(
            key=payload["receipt_sha256"], new_value=payload, domain="computation_formulations",
            cause="retain a unit-checked cross-domain recipe without promoting its semantic mapping"))

    @classmethod
    async def restore(cls, gateway: Any, identity: Any, catalog: Any) -> Any:
        payload = await gateway.read(identity, domain="computation_formulations", fresh=True)
        if payload is None:
            return None
        recipe = cls.from_dict(payload, catalog)
        if recipe.to_dict()["receipt_sha256"] != identity:
            raise ValueError("retained formulation identity differs")
        return recipe


@dataclass(frozen=True, slots=True)
class FormulationOutcome:
    formulation: Formulation
    context: KnowledgeContext
    completed: tuple[ComputedKnowledge, ...]
    gaps: tuple[dict, ...]

    def to_dict(self) -> dict[str, Any]:
        complete = not self.gaps
        body = {"schema": "aura.computation_formulation_outcome.v1",
                "formulation_receipt_sha256": self.formulation.to_dict()["receipt_sha256"],
                "scope": self.context.scope, "context_sha256": _sha(self.context.to_dict()),
                "completed": [item.to_dict() for item in self.completed], "gaps": self.gaps,
                "complete": complete, "all_consequences_hard": complete and all(
                    item.hard_constraint for item in self.completed),
                "semantic_definition": self.formulation.definition,
                "source_interpretation_proven": False, "general_transfer_proven": False}
        return {**body, "receipt_sha256": _sha(body)}


def run_formulation(formulation: Formulation, context: KnowledgeContext) -> FormulationOutcome:
    """Solve the available graph; preserve cycles and missing premises as gaps."""
    names = {cell.identity for cell in formulation.cells}
    if names & {item.identity for item in context.premises}:
        raise ValueError("formulation results cannot overwrite input evidence")
    if len(context.premises) + len(names) > 256:
        raise ValueError("formulation inputs and derived values exceed the context bound")
    pending, completed = list(formulation.cells), []
    working = context
    failures = {}
    while pending:
        advanced = False
        for cell in tuple(pending):
            required = set(dict(cell.bindings).values()) | set(dict(cell.assumption_bindings).values())
            if not required <= {item.identity for item in working.premises}:
                continue
            pending.remove(cell)
            try:
                result = compute_knowledge(cell, working)
            except (ValueError, KeyError, ArithmeticError) as exc:
                failures[cell.identity] = {"cell": cell.identity, "field": cell.model.field,
                    "status": "unestablished", "reason": f"{type(exc).__name__}:{exc}",
                    "required_premises": sorted(required), "reference": cell.model.reference}
                continue
            completed.append(result)
            working = replace(working, premises=(*working.premises, ScopedPremise(
                cell.identity, context.scope, result.result, "computed_knowledge",
                result.to_dict()["receipt_sha256"], "measurement" if result.hard_constraint else "estimate",
                dependencies=result.premise_ids)))
            advanced = True
        if not advanced:
            break
    for cell in pending:
        unavailable = sorted((set(dict(cell.bindings).values()) | set(dict(cell.assumption_bindings).values()))
                             - {item.identity for item in working.premises})
        failures[cell.identity] = {"cell": cell.identity, "field": cell.model.field,
            "status": "missing_or_cyclic", "required_premises": unavailable,
            "query": f"{cell.model.field}: {', '.join(unavailable)}; model {cell.model.identity}",
            "reference": cell.model.reference}
    return FormulationOutcome(formulation, context, tuple(completed),
                              tuple(failures[name] for name in sorted(failures)))


async def run_formulation_async(formulation: Any, context: Any) -> Any:
    from core.runtime.executors import off_the_loop
    return await off_the_loop(run_formulation, formulation, context)


async def retrieve_formulation_gaps(outcome: Any, providers: Any) -> tuple[Any, ...]:
    """Query available knowledge stores; passages never become numeric facts here."""
    import inspect

    from core.runtime.executors import off_the_loop
    rows = []
    for gap in outcome.gaps:
        query = gap.get("query", f"{gap['field']}: {gap['reason']}") if "reason" in gap else gap["query"]
        for name, provider in providers.items():
            try:
                value = await off_the_loop(provider, query)
                if inspect.isawaitable(value):
                    value = await value
                rows.append({"cell": gap["cell"], "provider": name, "query": query,
                             "retrieved": value, "premises_admitted": False})
            except Exception as exc:
                from core.runtime.errors import record_degradation
                record_degradation("computation_sandbox", exc, action="retain unresolved knowledge gap")
                rows.append({"cell": gap["cell"], "provider": name, "query": query,
                             "failure": f"{type(exc).__name__}:{exc}", "premises_admitted": False})
    return tuple(rows)


def solve_model_unknown(
    model: ComputationModel,
    *,
    unknown: str,
    known: dict,
    output: QuantityBounds,
) -> Any:
    """Solve a rational affine numerator and check the original equation and domain."""
    import sympy as sp
    units = dict(model.inputs)
    if (unknown not in units or set(known) != set(units) - {unknown}
            or not isinstance(output, QuantityBounds)
            or output.dimension != dimension_of(model.output_unit)
            or output.lower != output.upper):
        raise ValueError("inverse computation needs exact output and every other input")
    for name, value in known.items():
        if (not isinstance(value, QuantityBounds) or value.dimension != dimension_of(units[name])
                or value.lower != value.upper):
            raise ValueError("inverse computation needs exact dimensioned input values")
        if (name in model.nonnegative and value.lower < 0
                or name in model.positive and value.lower <= 0):
            raise ValueError("inverse input is outside the model's declared domain")
    symbol = sp.Symbol("unknown", real=True)
    values = {name: sp.Rational(value.lower.numerator, value.lower.denominator)
              for name, value in known.items()}
    values[unknown] = symbol

    def walk(node: Any) -> Any:
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
            number = _rational(node.value)
            return sp.Rational(number.numerator, number.denominator)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = walk(node.left), walk(node.right)
            operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                          ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
                          ast.Pow: lambda: left ** right}
            if type(node.op) in operations:
                return operations[type(node.op)]()
        raise ValueError("inverse expression is outside the reviewed arithmetic grammar")
    expression = walk(ast.parse(model.expression, mode="eval").body)
    target = sp.Rational(output.lower.numerator, output.lower.denominator)
    try:
        numerator, _denominator = sp.fraction(sp.together(expression - target))
        equation = sp.Poly(numerator, symbol)
    except sp.PolynomialError as exc:
        raise ValueError("inverse problem is outside the affine solver") from exc
    if equation.degree() != 1:
        raise ValueError("inverse problem is not a uniquely solvable affine equation")
    solutions = sp.linsolve([equation.as_expr()], [symbol])
    solution = next(iter(solutions))[0]
    if not solution.is_Rational or sp.simplify(expression.subs(symbol, solution) - target) != 0:
        raise ValueError("inverse solution failed exact substitution")
    result = QuantityBounds(_rational(str(solution)), _rational(str(solution)), dimension_of(units[unknown]))
    if (unknown in model.nonnegative and result.lower < 0
            or unknown in model.positive and result.lower <= 0):
        raise ValueError("inverse solution is outside the model's declared domain")
    if _expression(model.expression, {**known, unknown: result}) != output:
        raise ValueError("inverse solution failed dimensioned arithmetic replay")
    body = {"schema": "aura.computation_inverse.v1", "model_sha256": model.identity_sha256,
            "unknown": unknown, "known": {name: value.to_dict() for name, value in sorted(known.items())},
            "output": output.to_dict(), "solution": result.to_dict(),
            "exact_substitution_verified": True, "unestablished_assumptions": model.assumptions,
            "hard_constraint": False, "source_interpretation_proven": False}
    return result, {**body, "receipt_sha256": _sha(body)}
