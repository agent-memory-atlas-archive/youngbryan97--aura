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


def test_structured_observation_fills_a_gap_and_recomputes_the_actual_choice():
    portfolio, consequence = fixture()
    recipe = Formulation("inventory", "Balance after an observed removal.",
        (consequence.request,), (consequence.request.identity,))
    missing = replace(consequence.context, premises=consequence.context.premises[:1])
    calls = []
    async def inventory(required, context):
        calls.append((required, context.scope))
        return (consequence.context.premise("removed"),)
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=recipe, context=missing, observation_providers={"inventory": inventory}))
    assert calls == [(('removed',), missing.scope)]
    assert outcome.result == 28 and outcome.portfolio.decision.selected == "challenger"
    assert outcome.receipt["unresolved_gaps"] == 0
    assert outcome.receipt["observation_rounds"][0]["admitted"] == ["removed"]
    assert outcome.receipt["source_interpretation_proven"] is False


def test_observation_loop_continues_only_when_a_new_declared_input_arrives():
    portfolio, consequence = fixture()
    recipe = Formulation("inventory", "Balance needs two independently available observations.",
        (consequence.request,), (consequence.request.identity,))
    empty = replace(consequence.context, premises=())
    calls = []
    def one_at_a_time(required, context):
        calls.append(required)
        return (consequence.context.premise(required[0]),)
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=recipe, context=empty, observation_providers={"inventory": one_at_a_time}))
    assert calls == [("remaining", "removed"), ("removed",)]
    assert outcome.result == 28 and len(outcome.receipt["observation_rounds"]) == 2


def test_partial_graph_recomputation_reuses_only_observations_not_old_derived_values():
    portfolio, consequence = fixture()
    subtotal = ComputationRequest("subtotal", MODEL_CELLS["arithmetic_sum.v1"],
                                 (("a", "remaining"), ("b", "extra")))
    balance = replace(consequence.request, bindings=(("a", "subtotal"), ("b", "removed")))
    recipe = Formulation("inventory", "A measured subtotal precedes removal.",
                         (balance, subtotal), (balance.identity,))
    missing = replace(consequence.context, premises=(consequence.context.premises[0],
        replace(consequence.context.premises[0], identity="extra", value=QuantityBounds.of(0))))
    contexts = []
    def observe(required, context):
        contexts.append({item.identity for item in context.premises})
        assert required == ("removed",)
        return (consequence.context.premise("removed"),)
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=recipe, context=missing, observation_providers={"inventory": observe}))
    assert contexts == [{"remaining", "extra"}]
    assert outcome.result == 28 and outcome.receipt["unresolved_gaps"] == 0
    assert {item.request.identity for item in outcome.formulation.completed} == {"subtotal", "stock_balance"}
    assert len(outcome.receipt["observation_rounds"]) == 1


@pytest.mark.parametrize("fault", ["text", "other_scope", "stale", "future", "derived", "unrequested", "duplicate"])
def test_observation_ports_cannot_admit_untyped_foreign_or_unrequested_data(fault):
    portfolio, consequence = fixture()
    recipe = Formulation("inventory", "Balance needs fresh removal evidence.",
        (consequence.request,), (consequence.request.identity,))
    missing = replace(consequence.context, premises=consequence.context.premises[:1])
    item = consequence.context.premise("removed")
    payloads = {
        "text": ({"identity": "removed", "value": 4, "kind": "measurement"},),
        "other_scope": (replace(item, scope="other"),),
        "stale": (replace(item, valid_until=-1),),
        "future": (replace(item, valid_from=1),),
        "derived": (replace(item, dependencies=("remaining",)),),
        "unrequested": (replace(item, identity="remaining"),),
        "duplicate": (item, item),
    }
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=recipe, context=missing,
        observation_providers={"inventory": lambda _required, _context: payloads[fault]}))
    assert outcome.result == -28 and outcome.selection is None
    assert outcome.receipt["observation_rounds"][0]["attempts"][0]["failed"] is True
    assert outcome.receipt["observation_rounds"][0]["admitted"] == []


def test_conflicting_observation_ports_do_not_choose_the_first_store_as_truth():
    portfolio, consequence = fixture()
    recipe = Formulation("inventory", "Conflicting removal observations need resolution.",
        (consequence.request,), (consequence.request.identity,))
    missing = replace(consequence.context, premises=consequence.context.premises[:1])
    item = consequence.context.premise("removed")
    conflicting = replace(item, value=QuantityBounds.of(8), origin="sensor", ref="sensor:2")
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=recipe, context=missing, observation_providers={
            "a": lambda _required, _context: (item,),
            "b": lambda _required, _context: (conflicting,)}))
    assert outcome.result == -28 and outcome.selection is None
    record = outcome.receipt["observation_rounds"][0]
    assert record["conflicts"] == ["removed"] and record["admitted"] == []
    assert {row["provider"] for row in record["observations"]["removed"]} == {"a", "b"}


def test_an_observed_estimate_stays_conditional_after_gap_fill_and_recomputation():
    portfolio, consequence = fixture()
    recipe = Formulation("inventory", "Estimated removal is not an exact observation.",
        (consequence.request,), (consequence.request.identity,))
    missing = replace(consequence.context, premises=consequence.context.premises[:1])
    estimate = replace(consequence.context.premise("removed"), kind="estimate")
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=recipe, context=missing,
        observation_providers={"inventory": lambda _required, _context: (estimate,)}))
    assert outcome.result == -28 and outcome.receipt["unresolved_gaps"] == 0
    assert outcome.selection.receipt["hard_constraint_count"] == 0


def test_no_observation_progress_terminates_without_repeated_queries_or_a_timer():
    portfolio, consequence = fixture()
    recipe = Formulation("inventory", "An absent measurement remains absent.",
        (consequence.request,), (consequence.request.identity,))
    missing = replace(consequence.context, premises=consequence.context.premises[:1])
    calls = []
    def unavailable(required, context):
        calls.append(required)
        return ()
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=recipe, context=missing, observation_providers={"inventory": unavailable}))
    assert calls == [("removed",)] and outcome.result == -28
    assert outcome.receipt["unresolved_gaps"] == 1


def test_real_local_corpus_queries_flow_through_existing_intentional_memory(tmp_path):
    from core.knowledge.local_corpus import LocalCorpusStore
    from core.memory.intentional_retrieval import IntentionalRetriever, MemoryStoreType

    store = LocalCorpusStore(tmp_path / "corpus.db")
    store.add_documents([("Inventory accounting", "Removed stock changes the inventory balance.", "reference")])
    retriever = IntentionalRetriever()
    retriever.register_store(MemoryStoreType.REFERENCE,
        lambda query, limit: [hit.to_memory_dict() for hit in store.search(query, limit=limit)])
    portfolio, consequence = fixture()
    recipe = Formulation("inventory", "A reference explains removal but cannot measure current stock.",
        (consequence.request,), (consequence.request.identity,))
    missing = replace(consequence.context, premises=consequence.context.premises[:1])
    outcome = asyncio.run(run_semantic_computation_loop(portfolio=portfolio,
        formulation=recipe, context=missing, memory_retriever=retriever))
    assert outcome.result == -28 and outcome.selection is None
    retrieved = outcome.retrieved[0]["retrieved"]
    assert "reference" in retrieved["stores_queried"] and retrieved["hits"]
    # The router preserves the adapter envelope under its own metadata.
    source = retrieved["hits"][0]["metadata"]["metadata"]
    assert source["provenance"] == "local_corpus" and source["source"] == "reference"
    assert outcome.retrieved[0]["premises_admitted"] is False


def test_formulation_rejects_context_overflow_before_any_cell_runs():
    from core.reasoning.computation_sandbox import run_formulation
    _portfolio, consequence = fixture()
    recipe = Formulation("inventory", "A bounded calculation must reserve its output slots.",
        (consequence.request,), (consequence.request.identity,))
    extra = tuple(replace(consequence.context.premises[0], identity=f"extra:{index}") for index in range(254))
    full = replace(consequence.context, premises=(*consequence.context.premises, *extra))
    with pytest.raises(ValueError, match="context bound"):
        run_formulation(recipe, full)


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
