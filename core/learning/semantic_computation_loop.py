"""Connect calculation, candidate selection, knowledge gaps, and recipe retention."""

from dataclasses import dataclass, replace

from core.reasoning.computation_sandbox import (
    retrieve_formulation_gaps,
    run_formulation_async,
)
from core.reasoning.computational_knowledge import _sha


@dataclass(frozen=True, slots=True)
class SemanticComputationOutcome:
    portfolio: object
    formulation: object
    selection: object
    retrieved: tuple
    receipt: dict

    @property
    def result(self):
        if self.portfolio is None:
            return None
        return dict(self.portfolio.executions)[self.portfolio.decision.selected]["result"]


async def run_semantic_computation_loop(*, portfolio, formulation, context,
                                      providers=None, gateway=None, result_unit="count",
                                      selection_output=None):
    """One checked graph supplies evidence to the retained executable candidates.

    Source-to-equation mappings are caller proposals. Retrieval supplies cited
    material for those proposals; this function cannot admit passages as facts.
    An incomplete formulation leaves the existing selection untouched.
    """
    if context.scope != portfolio.source_sha256:
        raise ValueError("computation loop and semantic portfolio describe different problems")
    if selection_output is None:
        if len(formulation.outputs) != 1:
            raise ValueError("a multi-output formulation needs an explicit answer quantity")
        selection_output = formulation.outputs[0]
    if selection_output not in formulation.outputs:
        raise ValueError("answer quantity must be a declared formulation output")
    calculated = await run_formulation_async(formulation, context)
    completed = {item.request.identity: item for item in calculated.completed}
    selection = None
    updated = portfolio
    if not calculated.gaps and selection_output in completed:
        selection = await portfolio.reconcile_computed_constraints_async(
            (completed[selection_output],), now=context.now,
            result_unit=result_unit)
        updated = replace(portfolio, decision=selection.decision) if selection.decision is not None else None
    retrieved = await retrieve_formulation_gaps(calculated, providers or {})
    retained = None
    if gateway is not None:
        # Retain the equation graph and its unproved semantic mapping even when
        # it has gaps. A later context must recalculate all values and guards.
        retained = await formulation.retain(gateway)
    body = {"schema": "aura.semantic_computation_loop.v1", "source_sha256": context.scope,
            "formulation_outcome": calculated.to_dict(),
            "selection": None if selection is None else selection.receipt,
            "selection_output": selection_output,
            "selected": None if updated is None else updated.decision.selected,
            "retrieval_cells": [{"cell": row["cell"], "provider": row["provider"],
                                "query": row["query"], "premises_admitted": False,
                                "failed": "failure" in row} for row in retrieved],
            "retained_recipe_receipt_sha256": None if retained is None else formulation.to_dict()["receipt_sha256"],
            "unresolved_gaps": len(calculated.gaps), "source_interpretation_proven": False,
            "general_transfer_proven": False, "serving_authority": False}
    return SemanticComputationOutcome(updated, calculated, selection, retrieved,
                                      {**body, "receipt_sha256": _sha(body)})
