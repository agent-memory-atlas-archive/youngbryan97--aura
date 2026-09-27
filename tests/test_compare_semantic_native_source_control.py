"""A source-content intervention must be the only difference between arms."""

import pytest
from types import SimpleNamespace

from core.learning.procedure_induction import Instruction, Program

from tools.compare_semantic_native_source_control import (
    matched_source_control,
    matched_source_swap,
    matched_verification_identity,
)


def _arms():
    common = {"sources": ["a", "b"], "weight_mode": "fitted",
              "max_steps": 3, "checkpoint_receipt_sha256": "checkpoint"}
    full = {**common, "source_evidence": "source_text", "plan_sha256": "full-plan"}
    erased = {**common, "source_evidence": "source_token_erasure",
              "plan_sha256": "erased-plan"}
    full_report = {"plan_sha256": "full-plan", "population": 2,
                   "pair_exact": 1, "source_responsive": 1}
    erased_report = {"plan_sha256": "erased-plan", "population": 2,
                     "pair_exact": 0, "source_responsive": 0}
    full_rows = tuple({"source_sha256": source, "source_evidence": "source_text",
                       "program_equivalent": True, "answer_correct": True,
                       "program": {"steps": [source]}} for source in common["sources"])
    erased_rows = tuple({"source_sha256": source,
                         "source_evidence": "source_token_erasure",
                         "program_equivalent": source == "b", "answer_correct": source == "b",
                         "program": {"steps": ["a"]}}
                        for source in common["sources"])
    return full, erased, full_report, erased_report, full_rows, erased_rows


def test_adjudicator_counts_both_causal_changes_and_correctness():
    result = matched_source_control(*_arms())
    assert result["population"] == 2
    assert result["full_correct"] == 2 and result["erased_correct"] == 1
    assert result["full_only"] == 1 and result["erased_only"] == 0
    assert result["program_changed"] == 1
    assert result["full_source_responsive"] == 1
    assert result["erased_source_responsive"] == 0


def test_adjudicator_rejects_model_budget_population_and_mode_drift():
    full, erased, full_report, erased_report, full_rows, erased_rows = _arms()
    for changed in ({**erased, "max_steps": 4},
                    {**erased, "checkpoint_receipt_sha256": "other"},
                    {**erased, "source_evidence": "source_text"}):
        with pytest.raises(ValueError):
            matched_source_control(full, changed, full_report, erased_report,
                                   full_rows, erased_rows)
    with pytest.raises(ValueError, match="population"):
        matched_source_control(full, erased, full_report,
                               {**erased_report, "population": 1}, full_rows, erased_rows)
    with pytest.raises(ValueError, match="row coverage"):
        matched_source_control(full, erased, full_report, erased_report,
                               full_rows, erased_rows[::-1])


def test_paired_verification_rejects_implementation_and_checkpoint_drift():
    full = {"current_implementation_drift": [], "training_plan_sha256": "training",
            "checkpoint_receipt_sha256": "checkpoint", "source_evidence": "source_text"}
    erased = {**full, "source_evidence": "source_token_erasure"}
    matched_verification_identity(full, erased)
    for changed in ({**erased, "current_implementation_drift": ["source_control.py"]},
                    {**erased, "checkpoint_receipt_sha256": "other"},
                    {**erased, "source_evidence": "source_text"}):
        with pytest.raises(ValueError, match="verification identity"):
            matched_verification_identity(full, changed)
    matched_verification_identity(full, {**erased, "source_evidence": "source_pair_swap"},
                                  altered_mode="source_pair_swap")


def test_swap_is_graded_against_the_partner_after_generation():
    add = Program(2, (Instruction("add", (0, 1)),))
    sub = Program(2, (Instruction("sub", (0, 1)),))
    examples = (SimpleNamespace(inputs=(8, 3), program=add),
                SimpleNamespace(inputs=(8, 3), program=sub))
    common = {"sources": ["a", "b"], "source_pair_map": {"a": "b", "b": "a"},
              "weight_mode": "fitted", "max_steps": 3}
    full_plan = {**common, "source_evidence": "source_text", "plan_sha256": "full"}
    swap_plan = {**common, "source_evidence": "source_pair_swap", "plan_sha256": "swap"}
    full_report = {"plan_sha256": "full", "population": 2,
                   "pair_exact": 1, "source_responsive": 1}
    swap_report = {"plan_sha256": "swap", "population": 2,
                   "pair_exact": 0, "source_responsive": 0}
    full_rows = tuple({"source_sha256": identity, "scored_source_sha256": identity,
                       "source_evidence": "source_text", "program_equivalent": True,
                       "answer_correct": True, "decode_status": "completed",
                       "program": program.to_dict()}
                      for identity, program in zip(("a", "b"), (add, sub), strict=True))
    swap_rows = tuple({"source_sha256": identity, "scored_source_sha256": peer,
                       "source_evidence": "source_pair_swap", "program_equivalent": False,
                       "answer_correct": False, "decode_status": "completed",
                       "program": program.to_dict()}
                      for identity, peer, program in (("a", "b", sub), ("b", "a", add)))
    result = matched_source_swap(full_plan, swap_plan, full_report, swap_report,
                                 full_rows, swap_rows, examples)
    assert result["altered_original_correct"] == 0
    assert result["swapped_partner_correct"] == 2
    assert result["swapped_partner_answer_correct"] == 2
    assert result["swapped_matches_partner_full_program"] == 2
    assert result["exact_swap_pairs"] == 1
    with pytest.raises(ValueError, match="row coverage"):
        matched_source_swap(full_plan, swap_plan, full_report, swap_report,
                            full_rows, ({**swap_rows[0], "scored_source_sha256": "a"}, swap_rows[1]),
                            examples)
    with pytest.raises(ValueError, match="partner coverage"):
        matched_source_swap(full_plan, swap_plan, full_report, swap_report,
                            full_rows, swap_rows,
                            (examples[0], SimpleNamespace(inputs=(9, 3), program=sub)))
