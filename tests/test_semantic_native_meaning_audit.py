"""Denotation proofs do not excuse an altered requested procedure."""

import hashlib
from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent
from tools.verify_semantic_native_grammar import audit_grammar_meanings


def audit(target, generated, inputs):
    source = "The source-bound test request"
    example = SimpleNamespace(source_text=source, inputs=inputs, program=target, construction_id="test")
    row = {"source_sha256": hashlib.sha256(source.encode()).hexdigest(), "plan_sha256": "plan",
           "construction": "test", "target_available_to_scorer": False,
           "bound_forced_completion": False, "depth_bound_reached": False,
           "program": generated.to_dict(), "decode_status": "completed",
           "program_equivalent": semantic_programs_structurally_equivalent(target, generated),
           "answer_correct": target.run(inputs) == generated.run(inputs)}
    return audit_grammar_meanings([row], [example])


def test_associative_result_gets_a_proof_not_a_procedure_success():
    target = Program(3, (Instruction("add", (0, 1)), Instruction("add", (2, 3))))
    changed = Program(3, (Instruction("add", (1, 2)), Instruction("add", (3, 0))))
    result = audit(target, changed, (30, 6, 78))
    assert result["procedure_equivalent"] == 0
    assert result["proven_output_and_domain_equivalent"] == 1
    assert result["comparisons"][0]["meaning"]["method"] == "floor_integer_polynomial_v1"
    assert result["historical_totals_unchanged"] is True


def test_accidental_single_input_agreement_is_not_a_proof():
    result = audit(Program(2, (Instruction("add", (0, 1)),)),
                   Program(2, (Instruction("mul", (0, 1)),)), (2, 2))
    assert result["comparisons"][0]["observed_answer_correct"] is True
    assert result["proven_output_and_domain_equivalent"] == 0
    assert result["witnessed_different"] == 1


def test_partial_computation_cancellation_preserves_domain_obligations():
    partial = Program(2, (Instruction("idiv", (0, 1)), Instruction("sub", (2, 2))))
    total = Program(2, (Instruction("sub", (0, 0)),))
    result = audit(partial, total, (3, 2))
    assert result["proven_output_and_domain_equivalent"] == 0
    assert result["comparisons"][0]["meaning"]["distinction"] == "domain"


def test_equal_finite_observations_of_opaque_operators_stay_unknown():
    left = Program(1, (Instruction("head", (0,)),))
    # Use a supported opaque pair with identical finite observations but no floor proof.
    right = Program(1, (Instruction("reversed_", (0,)), Instruction("last", (1,))))
    result = audit(left, right, ((2, 3),))
    assert result["meaning_unknown"] == 1
    assert result["proven_output_and_domain_equivalent"] == 0


def test_missing_source_rows_are_not_empty_success():
    with pytest.raises(ValueError, match="complete matched"):
        audit_grammar_meanings([], [])
