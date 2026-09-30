"""Mixed source-grounded programs use one actual computation and selection path."""

import asyncio
from dataclasses import replace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_candidate_bank import SemanticCandidate, SemanticCandidateBank
from core.learning.semantic_candidate_union import GroundedProgram, unify_semantic_candidates
from core.learning.semantic_computation_loop import run_grounded_semantic_computation_loop
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import SemanticTransductionOutcome
from core.reasoning.computation_sandbox import Formulation
from core.reasoning.computational_knowledge import (
    ComputationRequest, KnowledgeContext, QuantityBounds, ScopedPremise,
)
from core.reasoning.computational_models import MODEL_CELLS


SOURCE = "a" * 64
INPUTS = (TokenSpan(0, 1), TokenSpan(2, 3))
WRONG = Program(2, (Instruction("sub", (0, 1)),))
RIGHT = Program(2, (Instruction("sub", (1, 0)),))


def bank(program):
    candidate = SemanticCandidate(program, 1., (TokenSpan(4, 5),), 0, 0)
    selected = SemanticTransductionOutcome(None, "unselected", {}, {})
    body = {"schema": "aura.semantic_candidate_bank.v3", "source_text_sha256": SOURCE,
        "candidates": [candidate.to_dict()], "input_spans": [span.to_dict() for span in INPUTS],
        "selected_program_sha256": None, "selected_refusal": "unselected",
        "selected_search_interrupted": False, "search_complete": False}
    return SemanticCandidateBank(selected, (candidate,), INPUTS,
                                 {**body, "receipt_sha256": _sha(body)})


def fixture():
    union = unify_semantic_candidates(banks={"bindings": bank(WRONG), "relations": bank(RIGHT)},
        additional={"native": GroundedProgram(RIGHT, INPUTS, SOURCE, "b" * 64)},
        public_inputs=(4, 32), source_sha256=SOURCE)
    context = KnowledgeContext(SOURCE, 0., (
        ScopedPremise("stock", SOURCE, QuantityBounds.of(32), "inventory", "inventory:1"),
        ScopedPremise("removed", SOURCE, QuantityBounds.of(4), "inventory", "inventory:2"),
    ))
    request = ComputationRequest("balance", MODEL_CELLS["arithmetic_difference.v1"],
                                 (("a", "stock"), ("b", "removed")))
    recipe = Formulation("inventory", "Balance after a scoped stock removal.", (request,), ("balance",))
    return union, context, recipe


def run(union, context, recipe, **options):
    return asyncio.run(run_grounded_semantic_computation_loop(candidates=union,
        public_inputs=(4, 32), incumbent_origin="bindings:0", formulation=recipe,
        context=context, **options))


def test_every_method_reaches_one_executed_portfolio_and_a_computed_choice():
    union, context, recipe = fixture()
    outcome = run(union, context, recipe)
    assert outcome.candidate_union == union
    assert len(outcome.portfolio.executions) == 2
    assert outcome.portfolio.selected_program == RIGHT and outcome.result == 28
    assert outcome.receipt["candidate_origins"][RIGHT.sha()] == ("relations:0", "native")
    assert outcome.receipt["candidate_union_receipt_sha256"] == union.receipt["receipt_sha256"]
    assert outcome.receipt["incumbent_program_sha256"] == WRONG.sha()
    assert outcome.receipt["initial_selected_program_sha256"] == WRONG.sha()
    assert outcome.receipt["method_agreement_used_as_correctness"] is False
    assert outcome.receipt["serving_authority"] is False
    assert outcome.receipt["general_transfer_proven"] is False


def test_method_agreement_does_not_replace_a_supported_incumbent_without_evidence():
    union, context, recipe = fixture()
    missing = replace(context, premises=context.premises[:1])
    outcome = run(union, missing, recipe)
    assert outcome.portfolio.selected_program == WRONG and outcome.result == -28
    assert outcome.selection is None and outcome.receipt["unresolved_gaps"] == 1
    assert len(outcome.receipt["candidate_origins"][RIGHT.sha()]) == 2


def test_observation_ports_recompute_the_choice_in_the_mixed_entry_point():
    union, context, recipe = fixture()
    missing = replace(context, premises=context.premises[:1])
    calls = []
    def inventory(required, current):
        calls.append((required, current.scope))
        return (context.premise("removed"),)
    outcome = run(union, missing, recipe, observation_providers={"inventory": inventory})
    assert calls == [(('removed',), SOURCE)]
    assert outcome.result == 28 and outcome.receipt["observation_rounds"][0]["admitted"] == ["removed"]


def test_a_computed_rejection_of_every_method_does_not_pick_the_least_wrong():
    union, context, recipe = fixture()
    changed = replace(context, premises=(replace(context.premises[0], value=QuantityBounds.of(64)),
                                         context.premises[1]))
    outcome = run(union, changed, recipe)
    assert outcome.portfolio is None and outcome.result is None
    assert outcome.selection.receipt["admissible"] == []
    assert outcome.candidate_union == union


@pytest.mark.parametrize("fault", ["scope", "receipt", "incumbent", "inputs"])
def test_unbound_or_modified_inputs_do_not_enter_mixed_computation(fault):
    union, context, recipe = fixture()
    options = dict(candidates=union, public_inputs=(4, 32), incumbent_origin="bindings:0",
                   formulation=recipe, context=context)
    if fault == "scope":
        options["context"] = replace(context, scope="other")
    elif fault == "receipt":
        options["candidates"] = replace(union, receipt={**union.receipt, "receipt_sha256": "c" * 64})
    elif fault == "incumbent":
        options["incumbent_origin"] = "absent"
    else:
        options["public_inputs"] = (4,)
    with pytest.raises(ValueError):
        asyncio.run(run_grounded_semantic_computation_loop(**options))


def test_union_conversion_is_a_floor_execution_not_a_cached_method_answer():
    union, _context, _recipe = fixture()
    first = union.to_portfolio(public_inputs=(4, 32), incumbent_origin="native")
    second = union.to_portfolio(public_inputs=(4, 64), incumbent_origin="native")
    assert dict(first.executions)[RIGHT.sha()]["result"] == 28
    assert dict(second.executions)[RIGHT.sha()]["result"] == 60
    assert first.selected_program == second.selected_program == RIGHT


def test_failed_incumbent_is_not_relabelled_as_the_executable_fallback():
    invalid = Program(2, (Instruction("length", (0,)),))
    _union, context, recipe = fixture()
    union = unify_semantic_candidates(banks={"bindings": bank(invalid), "relations": bank(RIGHT)},
        additional={}, public_inputs=(4, 32), source_sha256=SOURCE)
    outcome = run(union, context, recipe)
    assert not dict(outcome.portfolio.executions)[invalid.sha()]["completed"]
    assert outcome.portfolio.selected_program == RIGHT and outcome.result == 28
    assert outcome.receipt["incumbent_program_sha256"] == invalid.sha()
    assert outcome.receipt["initial_selected_program_sha256"] == RIGHT.sha()


def test_mixed_candidate_boundary_is_registered_with_its_executable_invariant():
    from core.organism.claims_computational_knowledge import install_computational_knowledge_claims
    from core.organism.model_validation import Evidence, Outcome, RuntimeModel, ValidationSuite

    suite = ValidationSuite()
    install_computational_knowledge_claims(suite)
    test = next(item for item in suite.tests() if item.name == "mixed_program_agreement_is_not_correctness")
    claim = next(item for item in suite.claims() if item.test == test.name)
    assert test.run(RuntimeModel().declare("scoped_computational_knowledge")).score.outcome is Outcome.PASS
    assert claim.evidence is Evidence.MEASURED_SYNTHETIC
    assert "no native interpretation" in claim.evidence_note
