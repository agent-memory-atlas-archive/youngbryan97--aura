"""A source occurrence is not a numerical value or a program vote."""

import asyncio
import json
from dataclasses import replace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_candidate_bank import SemanticCandidate, SemanticCandidateBank
from core.learning.semantic_meaning_hypothesis import (
    GroundedProgramProposal,
    MeaningBinding,
    MeaningOperation,
    MeaningStageInquiry,
    ObservedMeaningStageInquiry,
    SourceOccurrence,
    compare_meaning_stages,
    compare_source_bound_meanings,
    meaning_hypotheses_from_bank,
    plan_meaning_stage_inquiries,
    predict_meaning_trajectory,
    select_source_bound_portfolio,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer import SemanticTransductionOutcome


def _bank() -> SemanticCandidateBank:
    source = "a" * 64
    inputs = (TokenSpan(0, 1), TokenSpan(2, 3))
    candidate = SemanticCandidate(
        Program(2, (Instruction("sub", (0, 1)),)), 0.7,
        (TokenSpan(4, 5),), 0, 0,
        ((TokenSpan(1, 2), TokenSpan(3, 4)),),
        ((inputs[0], inputs[1]),), "optimizer_selected",
    )
    selected = SemanticTransductionOutcome(None, "test", {}, {})
    body = {"schema": "aura.semantic_candidate_bank.v3", "source_text_sha256": source,
            "candidates": [candidate.to_dict()],
            "input_spans": [span.to_dict() for span in inputs],
            "selected_program_sha256": None, "selected_refusal": "test",
            "selected_search_interrupted": False, "search_complete": False}
    return SemanticCandidateBank(selected, (candidate,), inputs,
                                 {**body, "receipt_sha256": _sha(body)})


def test_meaning_hypothesis_preserves_source_occurrences_and_compiles():
    bank = _bank()
    hypothesis, = bank.meaning_hypotheses()
    assert hypothesis.input_occurrences[0].identity != hypothesis.input_occurrences[1].identity
    assert hypothesis.operations[0].bindings[0].mention.identity != (
        hypothesis.operations[0].bindings[1].mention.identity)
    assert hypothesis.to_program() == bank.candidates[0].program
    grounded = GroundedProgramProposal(hypothesis, hypothesis.to_program())
    assert len(grounded.receipt_sha256) == 64


def test_role_intervention_changes_program_and_receipt():
    hypothesis, = meaning_hypotheses_from_bank(_bank())
    operation, = hypothesis.operations
    left, right = operation.bindings
    changed = replace(hypothesis, operations=(replace(operation, bindings=(
        replace(left, register=right.register), replace(right, register=left.register))),))
    assert changed.identity != hypothesis.identity
    assert changed.to_program().instructions[0].args == (1, 0)
    assert GroundedProgramProposal(changed, changed.to_program()).receipt_sha256 != (
        GroundedProgramProposal(hypothesis, hypothesis.to_program()).receipt_sha256)
    with pytest.raises(ValueError, match="differs"):
        GroundedProgramProposal(changed, hypothesis.to_program())


def test_candidate_bank_integrity_precedes_meaning_projection():
    bank = _bank()
    tampered = replace(bank, input_spans=(TokenSpan(0, 1), TokenSpan(3, 4)))
    with pytest.raises(ValueError, match="payload differs"):
        meaning_hypotheses_from_bank(tampered)


def test_missing_mention_evidence_does_not_become_a_meaning_hypothesis():
    bank = _bank()
    candidate = replace(bank.candidates[0], argument_spans=None, definition_spans=None,
                        definition_provenance="unavailable")
    body = {key: value for key, value in bank.receipt.items() if key != "receipt_sha256"}
    body["candidates"] = [candidate.to_dict()]
    changed = replace(bank, candidates=(candidate,), receipt={**body, "receipt_sha256": _sha(body)})
    assert meaning_hypotheses_from_bank(changed) == ()


def test_invalid_operation_or_score_cannot_enter_meaning_state():
    hypothesis, = _bank().meaning_hypotheses()
    with pytest.raises(ValueError, match="source or evidence"):
        replace(hypothesis, source_score=float("nan"))
    operation, = hypothesis.operations
    with pytest.raises(ValueError, match="invalid role structure"):
        replace(operation, name="unknown_primitive")


def test_equal_present_values_do_not_hide_a_counterfactual_role_distinction():
    hypothesis, = _bank().meaning_hypotheses()
    operation, = hypothesis.operations
    left, right = operation.bindings
    reversed_hypothesis = replace(hypothesis, operations=(replace(operation, bindings=(
        replace(left, register=right.register), replace(right, register=left.register))),))
    original = GroundedProgramProposal(hypothesis, hypothesis.to_program())
    reversed_proposal = GroundedProgramProposal(reversed_hypothesis, reversed_hypothesis.to_program())
    unresolved = compare_source_bound_meanings(original, reversed_proposal, (4, 4),
        counterfactual_count=0)
    assert unresolved["proposed_consequence_comparison"]["status"] == "unknown"
    compared = compare_source_bound_meanings(original, reversed_proposal, (4, 4),
        counterfactual_count=16)
    assert compared["proposed_consequence_comparison"]["status"] == "different"
    assert compared["proposed_consequence_comparison"]["witness"]["inputs"][0] != (
        compared["proposed_consequence_comparison"]["witness"]["inputs"][1]
    )
    assert compared["source_interpretation_status"] == "unresolved"
    assert compared["left_receipt_sha256"] != compared["right_receipt_sha256"]


def test_source_bound_comparison_rejects_cross_source_and_bad_geometry():
    hypothesis, = _bank().meaning_hypotheses()
    proposal = GroundedProgramProposal(hypothesis, hypothesis.to_program())
    with pytest.raises(ValueError, match="source bank"):
        other = replace(hypothesis, bank_receipt_sha256="b" * 64)
        compare_source_bound_meanings(proposal, GroundedProgramProposal(other, other.to_program()), (1, 2))
    with pytest.raises(ValueError, match="geometry"):
        compare_source_bound_meanings(proposal, proposal, (1,))


def test_source_bound_inquiry_needs_an_independent_observation():
    hypothesis, = _bank().meaning_hypotheses()
    operation, = hypothesis.operations
    left, right = operation.bindings
    reversed_hypothesis = replace(hypothesis, operations=(replace(operation, bindings=(
        replace(left, register=right.register), replace(right, register=left.register))),))
    original = GroundedProgramProposal(hypothesis, hypothesis.to_program())
    reversed_proposal = GroundedProgramProposal(reversed_hypothesis, reversed_hypothesis.to_program())
    portfolio = select_source_bound_portfolio((original, reversed_proposal), (4, 4),
        incumbent_receipt_sha256=original.receipt_sha256)
    assert portfolio.decision.selected == original.receipt_sha256
    inquiries = portfolio.plan_inquiries()
    assert inquiries
    inquiry = inquiries[0]
    assert inquiry.source_sha256 == hypothesis.source_text_sha256
    assert inquiry.to_dict()["observed_result"] is None
    assert inquiry.to_dict()["correctness_authority"] is False
    assert inquiry.compatible_methods(observed_result=original.program.run(inquiry.inputs)) == (
        original.receipt_sha256,
    )
    revised = portfolio.reconcile_inquiry(inquiry,
        observed_result=reversed_proposal.program.run(inquiry.inputs),
        origin="independent_test", ref="measurement-1")
    assert revised.selected == reversed_proposal.receipt_sha256


def test_meaning_trajectory_keeps_result_identity_separate_from_storage_position():
    hypothesis, = _bank().meaning_hypotheses()
    source = hypothesis.source_text_sha256
    second = MeaningOperation("add", SourceOccurrence(source, TokenSpan(5, 6)), (
        MeaningBinding(0, 2, SourceOccurrence(source, TokenSpan(6, 7)),
                       SourceOccurrence(source, TokenSpan(4, 5))),
        MeaningBinding(1, 1, SourceOccurrence(source, TokenSpan(7, 8)),
                       hypothesis.input_occurrences[1]),
    ))
    # sub(9, 4) = 5, then add(result:0, input:1) = 14.
    swapped = replace(hypothesis, operations=(replace(hypothesis.operations[0],
        bindings=tuple(replace(binding, register=1 - binding.register)
                       for binding in hypothesis.operations[0].bindings)), second))
    trajectory = predict_meaning_trajectory(swapped, (4, 9))
    assert tuple(step.result for step in trajectory.transitions) == (5, 14)
    assert tuple(identity.encode() for identity in trajectory.transitions[1].argument_identities) == (
        "result:0", "input:1")
    assert trajectory.transitions[1].argument_values == (5, 9)
    assert trajectory.transitions[1].execution_receipt["execution_engine"] == "universal_metered_floor"
    assert _bank().meaning_trajectories((4, 9))[0].result == -5


def test_stage_contrast_uses_source_operation_not_storage_position():
    hypothesis, = _bank().meaning_hypotheses()
    operation, = hypothesis.operations
    left, right = operation.bindings
    reversed_hypothesis = replace(hypothesis, operations=(replace(operation, bindings=(
        replace(left, register=right.register), replace(right, register=left.register))),))
    original = predict_meaning_trajectory(hypothesis, (4, 9))
    reversed_ = predict_meaning_trajectory(reversed_hypothesis, (4, 9))
    contrast, = compare_meaning_stages(original, reversed_)
    assert (contrast.left_result, contrast.right_result) == (-5, 5)
    assert tuple(item.encode() for item in contrast.left_arguments) == ("input:0", "input:1")
    assert tuple(item.encode() for item in contrast.right_arguments) == ("input:1", "input:0")
    assert contrast.reason == "binding_and_result"
    equal_contrast, = compare_meaning_stages(
        predict_meaning_trajectory(hypothesis, (4, 4)),
        predict_meaning_trajectory(reversed_hypothesis, (4, 4)))
    assert equal_contrast.reason == "binding"
    assert equal_contrast.left_result == equal_contrast.right_result == 0
    with pytest.raises(ValueError, match="source bank and input state"):
        compare_meaning_stages(original, predict_meaning_trajectory(reversed_hypothesis, (4, 8)))


def test_independent_stage_feedback_distinguishes_same_final_answer():
    hypothesis, = _bank().meaning_hypotheses()
    source = hypothesis.source_text_sha256
    third_input = SourceOccurrence(source, TokenSpan(8, 9))
    second = MeaningOperation("mul", SourceOccurrence(source, TokenSpan(9, 10)), (
        MeaningBinding(0, 3, SourceOccurrence(source, TokenSpan(10, 11)),
                       hypothesis.operations[0].occurrence),
        MeaningBinding(1, 2, SourceOccurrence(source, TokenSpan(11, 12)), third_input),
    ))
    first = replace(hypothesis, input_occurrences=(*hypothesis.input_occurrences, third_input),
                    operations=(*hypothesis.operations, second))
    original_operation = first.operations[0]
    reversed_operation = replace(original_operation, bindings=tuple(replace(
        binding, register=1 - binding.register) for binding in original_operation.bindings))
    reversed_ = replace(first, operations=(reversed_operation, second))
    proposals = tuple(GroundedProgramProposal(item, item.to_program())
                      for item in (first, reversed_))
    assert proposals[0].program.run((4, 9, 0)) == proposals[1].program.run((4, 9, 0)) == 0
    inquiry, = plan_meaning_stage_inquiries(proposals, ((4, 9, 0),))
    assert inquiry.operation == first.operations[0].occurrence
    assert dict(inquiry.predictions) == {
        proposals[0].receipt_sha256: -5,
        proposals[1].receipt_sha256: 5,
    }
    minimal, = plan_meaning_stage_inquiries(proposals,
        ((4, 9, 0), (7, 9, 0)), max_inquiries=1)
    assert minimal.inputs == (4, 9, 0)
    portfolio = select_source_bound_portfolio(proposals, (4, 9, 0),
        incumbent_receipt_sha256=proposals[0].receipt_sha256)
    assert portfolio.source_bound_proposals == proposals
    assert portfolio.plan_stage_inquiries()
    decision = portfolio.reconcile_stage_inquiries(((
        inquiry, 5, "environment:step_probe", "observation-1"),))
    assert decision is not None and decision.selected == proposals[1].receipt_sha256


def test_stage_inquiry_survives_gateway_reopen_and_replays_feedback(tmp_path):
    from core.state.state_gateway import ConcreteStateGateway

    hypothesis, = _bank().meaning_hypotheses()
    operation, = hypothesis.operations
    left, right = operation.bindings
    reversed_hypothesis = replace(hypothesis, operations=(replace(operation, bindings=(
        replace(left, register=right.register), replace(right, register=left.register))),))
    proposals = tuple(GroundedProgramProposal(item, item.to_program())
                      for item in (hypothesis, reversed_hypothesis))
    portfolio = select_source_bound_portfolio(proposals, (4, 4),
        incumbent_receipt_sha256=proposals[0].receipt_sha256)
    inquiry = portfolio.plan_stage_inquiries()[0]
    assert MeaningStageInquiry.from_dict(json.loads(json.dumps(inquiry.to_dict()))) == inquiry
    tampered = inquiry.to_dict()
    tampered["predictions"][0][1] = 999
    with pytest.raises(ValueError, match="content or source binding"):
        MeaningStageInquiry.from_dict(tampered)
    with pytest.raises(ValueError, match="exact semantic algebra"):
        replace(inquiry, inputs=(True, *inquiry.inputs[1:]))
    with pytest.raises(ValueError, match="source-bound predictions"):
        replace(inquiry, operation=None)

    async def run():
        gateway = ConcreteStateGateway(root=tmp_path, governance_decide=lambda **_: True)
        receipts = await portfolio.retain_stage_inquiries(gateway)
        assert inquiry.identity in {receipt.key for receipt in receipts}
        reopened = ConcreteStateGateway(root=tmp_path, governance_decide=lambda **_: True)
        restored = await MeaningStageInquiry.restore(reopened, inquiry.identity)
        assert restored == inquiry
        observed = ObservedMeaningStageInquiry(
            restored, dict(restored.predictions)[proposals[1].receipt_sha256],
            "source:step", "measurement-1")
        await observed.retain(reopened)
        again = ConcreteStateGateway(root=tmp_path)
        assert await ObservedMeaningStageInquiry.restore_all(again) == (observed,)
        decision = await portfolio.reconcile_retained_stage_inquiries(again)
        assert decision is not None and decision.selected == proposals[1].receipt_sha256
        assert portfolio.decision.selected == proposals[0].receipt_sha256

    asyncio.run(run())
    assert portfolio.decision.selected == proposals[0].receipt_sha256
    assert portfolio.reconcile_stage_inquiries(((
        inquiry, 50, "environment:step_probe", "observation-2"),)) is None
    with pytest.raises(ValueError, match="contradict itself"):
        portfolio.reconcile_stage_inquiries(((
            inquiry, 5, "environment:step_probe", "observation-1"),
            (inquiry, -5, "environment:step_probe", "observation-1")))
    tampered = replace(inquiry, predictions=tuple(reversed(inquiry.predictions)))
    with pytest.raises(ValueError, match="differs from source-bound execution"):
        portfolio.reconcile_stage_inquiries(((
            tampered, 5, "environment:step_probe", "observation-1"),))
