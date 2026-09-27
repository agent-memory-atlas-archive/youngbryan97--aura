"""A matched fit comparison keeps unknowns, baselines, and program identity."""

from copy import deepcopy

import pytest

from core.learning.semantic_native_source_control import SOURCE_ERASURE_CONTRACT
from tools.verify_semantic_native_fit import compare_source_erasure_outcomes, verify_native_totals


def fixture():
    reference = {"schema": "aura.semantic_native_fit_plan.v1", "plan_sha256": "reference",
                 "held_ids": ["a", "b"], "steps": 320, "seed": 7, "implementation": {"fit": "old"}}
    control = {**reference, "schema": "aura.semantic_native_fit_plan.v2", "plan_sha256": "control",
               "source_evidence_control": SOURCE_ERASURE_CONTRACT, "implementation": {"fit": "new"}}
    rows = [{"source": source, "incumbent_correct": False, "pretrained_correct": False,
             "pretrained_program_sha256": "base", "bank_reachable": True,
             "program_sha256s": ["x", "y"], "chosen_program_sha256": "x", "selected_correct": success}
            for source, success in (("a", True), ("b", False))]

    def report(plan):
        rows_copy = deepcopy(rows)
        result = {"rows": rows_copy, "population": 2, "incumbent_correct": 0, "native_correct": 1,
                  "pretrained_correct": 0, "bank_reachable": 2, "gains": 1, "regressions": 0,
                  "learning_gains": 1, "learning_regressions": 0, "plan_sha256": plan["plan_sha256"],
                  "receipt_sha256": plan["plan_sha256"] + "-report"}
        verify_native_totals(result, rows_copy)
        return result
    return reference, control, report(reference), report(control)


def test_tied_accuracy_is_not_tied_program_selection():
    arguments = fixture()
    arguments[3]["rows"][0]["chosen_program_sha256"] = "y"
    result = compare_source_erasure_outcomes(*arguments)
    assert result["both_correct"] == result["both_incorrect"] == 1
    assert result["different_selected_programs"] == 1
    assert result["paired_exact_test"]["discordant"] == 0
    assert result["paired_exact_test"]["one_sided_exact_p"] == 1


@pytest.mark.parametrize("defect", ["steps", "seed", "population", "duplicate", "plan", "baseline", "bank", "relabel"])
def test_comparison_rejects_unmatched_protocol_or_observations(defect):
    reference, control, left, right = fixture()
    if defect in {"steps", "seed"}:
        control[defect] += 1
    elif defect == "population":
        control["held_ids"] = ["a"]
    elif defect == "duplicate":
        right["rows"][1]["source"] = "a"
    elif defect == "plan":
        right["plan_sha256"] = "other"
    elif defect == "baseline":
        right["rows"][0]["pretrained_program_sha256"] = "other"
    elif defect == "bank":
        right["rows"][0]["program_sha256s"] = ["x"]
    else:
        control["schema"] = "aura.semantic_native_fit_plan.v1"
    with pytest.raises(ValueError):
        compare_source_erasure_outcomes(reference, control, left, right)


def test_unknowns_do_not_become_paired_failures():
    reference, control, left, right = fixture()
    right["rows"][1]["selected_correct"] = None
    result = compare_source_erasure_outcomes(reference, control, left, right)
    assert result["known_pairs"] == result["unknown_pairs"] == 1
    assert result["both_incorrect"] == 0
