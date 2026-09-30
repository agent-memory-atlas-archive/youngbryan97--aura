"""Derive scoped facts with the existing proof kernel and explicit premise ancestry."""

from core.reasoning.computational_knowledge import ScopedPremise, _sha, premise_closure
from core.reasoning.natural_deduction import Bot, Not, formula_to_dict, parse, prove
from core.reasoning.proof_kernel import check_proof
from typing import Any


def derive_scoped_fact(
    context: Any,
    premise_ids: tuple[Any, ...],
    *,
    identity: Any,
    goal: Any,
) -> tuple[Any, dict[str, Any]]:
    """A checked conditional theorem can fill a model guard; inconsistency cannot."""
    premise_ids = tuple(premise_ids)
    if (not premise_ids or len(premise_ids) > 16 or len(set(premise_ids)) != len(premise_ids)
            or not isinstance(goal, str) or not 0 < len(goal) <= 256):
        raise ValueError("logical cell requires bounded, distinct premises and goal")
    ancestry = premise_closure(context, premise_ids)
    formulas = []
    for name in premise_ids:
        item = ancestry[name]
        if type(item.value) is not bool or len(item.proposition) > 256:
            raise ValueError("logical cell premises must be explicit bounded propositions")
        formula = parse(item.proposition)
        formulas.append(formula if item.value else Not(formula))
    target = parse(goal)
    encodings = [formula_to_dict(item) for item in (*formulas, target)]
    atoms = set()
    nodes = 0

    def inspect(node: Any) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > 128:
            raise ValueError("logical cell exceeds its expression bound")
        if node["t"] == "atom":
            atoms.add(node["name"])
        for value in node.values():
            if isinstance(value, dict):
                inspect(value)
    for encoding in encodings:
        inspect(encoding)
    if len(atoms) > 8:
        raise ValueError("logical cell exceeds its eight-atom search bound")
    consistency = prove(formulas, Bot())
    if consistency.provable:
        verdict = check_proof(formulas, Bot(), consistency.certificate)
        if not verdict.verified:
            raise ValueError("logical inconsistency proof was rejected by the kernel")
        raise ValueError("contradictory premises cannot justify a knowledge fact")
    proof = prove(formulas, target)
    body = {"schema": "aura.scoped_logical_computation.v1", "scope": context.scope,
            "premise_ids": sorted(ancestry), "goal": goal, "provable": proof.provable,
            "premise_encodings": encodings[:-1], "goal_encoding": encodings[-1],
            "hard_constraint": False, "source_interpretation_proven": False}
    fact = None
    if proof.provable:
        verdict = check_proof(formulas, target, proof.certificate)
        if not verdict.verified:
            raise ValueError("logical consequence was rejected by the kernel")
        hard = all(item.kind in {"given", "measurement"} for item in ancestry.values())
        body.update(kernel_verdict=verdict.to_dict(), certificate=proof.certificate.to_dict(),
                    hard_constraint=hard)
        fact = ScopedPremise(identity, context.scope, True, "proof_kernel", _sha(body),
                             "measurement" if hard else "assumption", goal,
                             dependencies=tuple(sorted(ancestry)))
    else:
        body["countermodel"] = proof.countermodel
    return fact, {**body, "receipt_sha256": _sha(body)}
