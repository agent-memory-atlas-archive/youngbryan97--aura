"""Connect calculation, candidate selection, knowledge gaps, and recipe retention."""

from dataclasses import dataclass, replace

from core.reasoning.computation_sandbox import (
    retrieve_formulation_gaps,
    run_formulation_async,
)
from core.reasoning.computational_knowledge import _sha
from typing import Any


@dataclass(frozen=True, slots=True)
class SemanticComputationOutcome:
    portfolio: object
    formulation: object
    selection: object
    retrieved: tuple
    receipt: dict
    candidate_union: object = None

    @property
    def result(self) -> Any:
        if self.portfolio is None:
            return None
        return dict(self.portfolio.executions)[self.portfolio.decision.selected]["result"]


async def _observe_missing_premises(outcome: Any, providers: Any) -> tuple[Any, dict[str, Any]]:
    """Caller-bound observation ports can supply data; text retrieval cannot."""
    import inspect

    from core.reasoning.computational_knowledge import KnowledgeContext, ScopedPremise
    from core.runtime.errors import record_degradation
    from core.runtime.executors import off_the_loop

    context = outcome.context
    cells = {cell.identity for cell in outcome.formulation.cells}
    required = tuple(sorted({identity for gap in outcome.gaps
        for identity in gap["required_premises"] if identity not in cells}
        - {premise.identity for premise in context.premises}))
    candidates, attempts = {}, []
    if required:
        for name, provider in sorted(providers.items()):
            try:
                observations = await off_the_loop(provider, required, context)
                if inspect.isawaitable(observations):
                    observations = await observations
                if (not isinstance(observations, tuple) or len(observations) > len(required)
                        or any(not isinstance(item, ScopedPremise) for item in observations)
                        or len({item.identity for item in observations}) != len(observations)
                        or any(item.identity not in required or item.dependencies for item in observations)):
                    raise ValueError("observation port returned unrequested, derived, or untyped evidence")
                checked = KnowledgeContext(context.scope, context.now, observations)
                for observation in observations:
                    checked.premise(observation.identity)
                for observation in observations:
                    candidates.setdefault(observation.identity, []).append((name, observation))
                attempts.append({"provider": name, "returned": len(observations), "failed": False})
            except Exception as exc:
                record_degradation("semantic_computation_loop", exc,
                    action="keep unresolved computation evidence")
                attempts.append({"provider": name, "failed": True,
                                 "reason": f"{type(exc).__name__}:{exc}"})
    admitted, conflicts = [], []
    for identity, rows in sorted(candidates.items()):
        first = rows[0][1]
        # Disagreement remains a gap. There is no first-store-wins authority.
        if any((item.value, item.kind, item.proposition, item.valid_from, item.valid_until)
                != (first.value, first.kind, first.proposition, first.valid_from, first.valid_until)
                for _name, item in rows[1:]):
            conflicts.append(identity)
            continue
        admitted.append(first)
    if len(context.premises) + len(admitted) + len(cells) > 256:
        raise ValueError("observed formulation would exceed the context bound")
    updated = replace(context, premises=(*context.premises, *admitted))
    body = {"schema": "aura.computation_observation_round.v1", "scope": context.scope,
            "requested": required, "attempts": attempts, "conflicts": conflicts,
            "admitted": [item.identity for item in admitted],
            "observations": {identity: [{"provider": name, "premise": item.to_dict()}
                for name, item in rows] for identity, rows in sorted(candidates.items())},
            "before_context_sha256": _sha(context.to_dict()),
            "after_context_sha256": _sha(updated.to_dict()),
            "provider_authority": "caller_bound_observation_port",
            "retrieved_text_admitted": False, "source_interpretation_proven": False}
    return updated, {**body, "receipt_sha256": _sha(body)}


async def run_semantic_computation_loop(
    *,
    portfolio: Any,
    formulation: Any,
    context: Any,
    providers: Any=None,
    gateway: Any=None,
    result_unit: str='count',
    selection_output: Any=None,
    observation_providers: Any=None,
    memory_retriever: Any=None,
) -> Any:
    """One checked graph supplies evidence to the retained executable candidates.

    Source-to-equation mappings are caller proposals. Retrieval supplies cited
    material for those proposals; this function cannot admit passages as facts.
    Explicit observation ports may supply fresh, scoped input data. Recompute
    after progress; every pass adds a previously absent declared input, so the
    loop terminates without a timer. An incomplete graph preserves selection.
    """
    if context.scope != portfolio.source_sha256:
        raise ValueError("computation loop and semantic portfolio describe different problems")
    if selection_output is None:
        if len(formulation.outputs) != 1:
            raise ValueError("a multi-output formulation needs an explicit answer quantity")
        selection_output = formulation.outputs[0]
    if selection_output not in formulation.outputs:
        raise ValueError("answer quantity must be a declared formulation output")
    initial_context_sha256 = _sha(context.to_dict())
    calculated = await run_formulation_async(formulation, context)
    observation_rounds = []
    while calculated.gaps and observation_providers:
        updated_context, observation = await _observe_missing_premises(calculated, observation_providers)
        observation_rounds.append(observation)
        if updated_context == context:
            break
        context = updated_context
        calculated = await run_formulation_async(formulation, context)
    completed = {item.request.identity: item for item in calculated.completed}
    selection = None
    updated = portfolio
    if not calculated.gaps and selection_output in completed:
        selection = await portfolio.reconcile_computed_constraints_async(
            (completed[selection_output],), now=context.now,
            result_unit=result_unit)
        updated = replace(portfolio, decision=selection.decision) if selection.decision is not None else None
    retrieval_providers = dict(providers or {})
    if memory_retriever is not None:
        if "intentional_memory" in retrieval_providers:
            raise ValueError("intentional memory provider is already bound")
        from core.memory.intentional_retrieval import RetrievalIntent

        def retrieve(query: Any) -> Any:
            return memory_retriever.retrieve(RetrievalIntent(
                task=query, query=query, kind="learn", whose_values="", limit=8)).to_dict()
        retrieval_providers["intentional_memory"] = retrieve
    retrieved = await retrieve_formulation_gaps(calculated, retrieval_providers)
    retained = None
    if gateway is not None:
        # Retain the equation graph and its unproved semantic mapping even when
        # it has gaps. A later context must recalculate all values and guards.
        retained = await formulation.retain(gateway)
    body = {"schema": "aura.semantic_computation_loop.v2", "source_sha256": context.scope,
            "initial_context_sha256": initial_context_sha256,
            "observation_rounds": observation_rounds,
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


async def run_grounded_semantic_computation_loop(
    *,
    candidates: Any,
    public_inputs: Any,
    incumbent_origin: Any,
    formulation: Any,
    context: Any,
    fuel: int=2000000,
    providers: Any=None,
    gateway: Any=None,
    result_unit: str='count',
    selection_output: Any=None,
    observation_providers: Any=None,
    memory_retriever: Any=None,
) -> Any:
    """Connect the mixed candidate union to the same scoped computation loop.

    Callers supply source-grounded proposals and a proposed equation mapping,
    never expected answers. All distinct programs execute in common input
    coordinates. Equal programs retain every origin but receive no voting bonus.
    Neither this entry point nor its evidence grants live serving authority.
    """
    from core.learning.semantic_candidate_union import SemanticCandidateUnion
    from core.runtime.executors import off_the_loop

    if not isinstance(candidates, SemanticCandidateUnion) or candidates.source_sha256 != context.scope:
        raise ValueError("mixed computations need a candidate union for the same problem")
    portfolio = await off_the_loop(candidates.to_portfolio, public_inputs=public_inputs,
                                  incumbent_origin=incumbent_origin, fuel=fuel)
    outcome = await run_semantic_computation_loop(portfolio=portfolio,
        formulation=formulation, context=context, providers=providers, gateway=gateway,
        result_unit=result_unit, selection_output=selection_output,
        observation_providers=observation_providers, memory_retriever=memory_retriever)
    body = {key: value for key, value in outcome.receipt.items() if key != "receipt_sha256"}
    body.update(schema="aura.semantic_computation_loop.v3",
        candidate_union_receipt_sha256=candidates.receipt["receipt_sha256"],
        candidate_origins={row.program.sha(): row.origins for row in candidates.candidates},
        incumbent_origin=incumbent_origin,
        incumbent_program_sha256=next(row.program.sha() for row in candidates.candidates
                                      if incumbent_origin in row.origins),
        initial_selected_program_sha256=portfolio.selected_program.sha() if portfolio.selected_program else None,
        method_agreement_used_as_correctness=False)
    return replace(outcome, candidate_union=candidates,
                   receipt={**body, "receipt_sha256": _sha(body)})
