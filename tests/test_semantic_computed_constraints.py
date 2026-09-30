"""The portfolio API consumes scoped consequences without answer keys."""

import asyncio
from dataclasses import replace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_computation_loop import run_semantic_computation_loop
from core.learning.semantic_program_portfolio import select_semantic_program_portfolio
from core.reasoning.computation_sandbox import Formulation
from core.reasoning.computational_knowledge import (
    ComputationRequest,
    KnowledgeContext,
    QuantityBounds,
    ScopedPremise,
    compute_knowledge,
)
from core.reasoning.computational_models import MODEL_CELLS


def fixture(*, kind="given", a=4, b=32):
    scope = "a" * 64
    wrong, right = Program(2, (Instruction("sub", (0, 1)),)), Program(2, (Instruction("sub", (1, 0)),))
    portfolio = select_semantic_program_portfolio(proposals={"incumbent": wrong, "challenger": right},
        public_inputs=(4, 32), incumbent="incumbent", observation_sha256=scope,
        provenance={"incumbent": "b" * 64, "challenger": "c" * 64})
    context = KnowledgeContext(scope, 0., (
        ScopedPremise("remaining", scope, QuantityBounds.of(b), "inventory", "inventory:1", kind),
        ScopedPremise("removed", scope, QuantityBounds.of(a), "inventory", "inventory:2", kind),
    ))
    computation = compute_knowledge(ComputationRequest("stock_balance", MODEL_CELLS["arithmetic_difference.v1"],
        (("a", "remaining"), ("b", "removed"))), context)
    return portfolio, computation


def test_computed_consequence_actually_changes_the_existing_candidate_decision():
    portfolio, consequence = fixture()
    selected = portfolio.reconcile_computed_constraints((consequence,), now=0.)
    assert portfolio.decision.selected == "incumbent"
    assert selected.decision.selected == "challenger"
    assert selected.receipt["admissible"] == ["challenger"]
    assert selected.receipt["source_interpretation_proven"] is False


def test_estimate_is_visible_but_cannot_override_the_incumbent_as_fact():
    portfolio, consequence = fixture(kind="estimate")
    selected = portfolio.reconcile_computed_constraints((consequence,), now=0.)
    assert selected.decision.selected == "incumbent" and selected.receipt["hard_constraint_count"] == 0
    assert any(not item["overlaps"] for item in selected.receipt["rows"][0]["comparisons"])


def test_refuted_candidate_set_returns_no_answer_instead_of_choosing_the_least_wrong():
    portfolio, consequence = fixture(a=4, b=64)
    selected = portfolio.reconcile_computed_constraints((consequence,), now=0.)
    assert selected.decision is None and selected.receipt["admissible"] == []


def test_cross_request_or_tampered_knowledge_never_reaches_the_selector():
    portfolio, consequence = fixture()
    other = replace(consequence, context=replace(consequence.context, scope="other"))
    with pytest.raises(ValueError, match="another problem"):
        portfolio.reconcile_computed_constraints((other,), now=0.)
    with pytest.raises(ValueError, match="replay"):
        portfolio.reconcile_computed_constraints((replace(consequence, result=QuantityBounds.of(-28)),), now=0.)


class Gateway:
    def __init__(self):
        self.rows = {}

    async def mutate(self, request):
        self.rows[request.domain, request.key] = request.new_value
        return request

    async def read(self, identity, *, domain, fresh):
        assert fresh
        return self.rows.get((domain, identity))


def test_closed_loop_computes_selects_retains_restores_and_recomputes_with_new_facts():
    portfolio, consequence = fixture()
    formulation = Formulation("inventory", "Remaining stock after a measured removal.",
                              (consequence.request,), (consequence.request.identity,))
    gateway = Gateway()
    first = asyncio.run(run_semantic_computation_loop(portfolio=portfolio, formulation=formulation,
                        context=consequence.context, gateway=gateway))
    assert first.portfolio.selected_program == dict(portfolio.proposals)["challenger"]
    assert first.result == 28 and first.receipt["unresolved_gaps"] == 0
    restored = asyncio.run(Formulation.restore(gateway, formulation.to_dict()["receipt_sha256"], MODEL_CELLS))
    assert restored == formulation
    # A different observed direction reverses the choice, without a cached answer.
    opposite = replace(consequence.context, premises=tuple(
        replace(item, value=QuantityBounds.of(4 if item.identity == "remaining" else 32))
        for item in consequence.context.premises))
    second = asyncio.run(run_semantic_computation_loop(portfolio=portfolio, formulation=restored, context=opposite))
    assert second.result == -28 and second.portfolio.decision.selected == "incumbent"


def test_ablation_leaves_the_original_choice_and_sends_real_gaps_to_a_provider():
    portfolio, consequence = fixture()
    formulation = Formulation("inventory", "Remaining stock after removal.",
                              (consequence.request,), (consequence.request.identity,))
    missing = replace(consequence.context, premises=consequence.context.premises[:1])
    queries = []
    async def memory(query):
        queries.append(query)
        return [{"ref": "episode:1", "text": "prior stock observation; not current evidence"}]
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio, formulation=formulation,
        context=missing, providers={"episodic_memory": memory}))
    assert outcome.result == -28 and outcome.selection is None
    assert queries and outcome.receipt["unresolved_gaps"] == 1
    assert outcome.retrieved[0]["premises_admitted"] is False


def test_multiple_output_quantities_require_an_explicit_binding_to_the_answer():
    portfolio, consequence = fixture()
    total = ComputationRequest("total", MODEL_CELLS["arithmetic_sum.v1"],
                               consequence.request.bindings)
    formulation = Formulation("inventory", "Balance and total are different quantities.",
        (consequence.request, total), (consequence.request.identity, total.identity))
    with pytest.raises(ValueError, match="explicit answer quantity"):
        asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
            formulation=formulation, context=consequence.context))
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=formulation, context=consequence.context, selection_output="stock_balance"))
    assert outcome.result == 28
    assert len(outcome.formulation.completed) == 2
    assert outcome.receipt["selection_output"] == "stock_balance"
    with pytest.raises(ValueError, match="declared formulation output"):
        asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
            formulation=formulation, context=consequence.context, selection_output="missing"))


def test_an_unresolved_cell_cannot_be_hidden_by_a_completed_answer_output():
    portfolio, consequence = fixture()
    missing = ComputationRequest("unresolved", MODEL_CELLS["arithmetic_sum.v1"],
                                (("a", "remaining"), ("b", "unknown")))
    formulation = Formulation("inventory", "Balance requires a separate observation.",
                              (consequence.request, missing), (consequence.request.identity,))
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=formulation, context=consequence.context))
    assert outcome.selection is None and outcome.result == -28
    assert outcome.receipt["unresolved_gaps"] == 1


def test_provider_failure_is_recorded_without_promoting_or_discarding_the_answer():
    portfolio, consequence = fixture()
    formulation = Formulation("inventory", "Balance requires removal evidence.",
                              (consequence.request,), (consequence.request.identity,))
    missing = replace(consequence.context, premises=consequence.context.premises[:1])
    def unavailable(_query):
        raise OSError("offline corpus unavailable")
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=formulation, context=missing, providers={"corpus": unavailable}))
    assert outcome.result == -28 and outcome.selection is None
    assert outcome.receipt["retrieval_cells"][0]["failed"] is True
    assert outcome.retrieved[0]["premises_admitted"] is False


def test_recipe_survives_a_real_state_gateway_reopen_without_retaining_an_answer(tmp_path):
    from core.state.state_gateway import ConcreteStateGateway

    portfolio, consequence = fixture()
    formulation = Formulation("inventory", "Remaining stock after a measured removal.",
                              (consequence.request,), (consequence.request.identity,))
    async def run():
        root = tmp_path / "state"
        gateway = ConcreteStateGateway(root=root, governance_decide=lambda **_k: {
            "approved": True, "receipt_id": "test-recipe-retention"})
        first = await run_semantic_computation_loop(portfolio=portfolio,
            formulation=formulation, context=consequence.context, gateway=gateway)
        key = first.receipt["retained_recipe_receipt_sha256"]
        reopened = ConcreteStateGateway(root=root)
        restored = await Formulation.restore(reopened, key, MODEL_CELLS)
        assert restored == formulation
        payload = await reopened.read(key, domain="computation_formulations", fresh=True)
        assert payload["retained_answers_available"] is False
        assert "result" not in payload and "context" not in payload
        second = await run_semantic_computation_loop(portfolio=portfolio,
            formulation=restored, context=consequence.context)
        assert second.result == 28
    asyncio.run(run())
