"""A durable grammar receipt is not an authority over its own answer."""

from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from tools.verify_semantic_native_grammar import verify_grammar_row


def test_grammar_row_reexecutes_and_rejects_forged_success():
    target = Program(2, (Instruction("sub", (0, 1)),))
    rival = Program(2, (Instruction("sub", (1, 0)),))
    example = SimpleNamespace(inputs=(8, 3), program=target, construction_id="novel")
    row = {"source_sha256": "source", "plan_sha256": "plan", "construction": "novel",
           "target_available_to_scorer": False, "program": target.to_dict(),
           "decode_status": "completed", "program_equivalent": True, "answer_correct": True,
           "bound_forced_completion": False, "depth_bound_reached": False}
    assert verify_grammar_row(row, example=example, identity="source", plan_sha256="plan") == (True, True)
    with pytest.raises(ValueError, match="independent execution"):
        verify_grammar_row({**row, "program": rival.to_dict()}, example=example,
                           identity="source", plan_sha256="plan")
    with pytest.raises(ValueError, match="target-blind"):
        verify_grammar_row({**row, "target_available_to_scorer": True}, example=example,
                           identity="source", plan_sha256="plan")
    with pytest.raises(ValueError, match="depth-bound"):
        verify_grammar_row({**row, "depth_bound_reached": True}, example=example,
                           identity="source", plan_sha256="plan")


def test_grammar_row_keeps_incomplete_programs_out_of_success_counts():
    program = Program(2, (Instruction("add", (0, 1)),))
    example = SimpleNamespace(inputs=(8, 3), program=program, construction_id="novel")
    row = {"source_sha256": "source", "plan_sha256": "plan", "construction": "novel",
           "target_available_to_scorer": False, "program": program.to_dict(),
           "decode_status": "disconnected_at_depth_bound", "program_equivalent": False,
           "answer_correct": False, "bound_forced_completion": False,
           "depth_bound_reached": True}
    assert verify_grammar_row(row, example=example, identity="source", plan_sha256="plan") == (False, False)
    with pytest.raises(ValueError, match="independent execution"):
        verify_grammar_row({**row, "answer_correct": True}, example=example,
                           identity="source", plan_sha256="plan")
