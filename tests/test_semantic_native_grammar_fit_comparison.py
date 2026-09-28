"""Generated fit claims require identical source/scoring protocols and honest unknowns."""

from copy import deepcopy

import pytest

from tools.compare_semantic_native_grammar_fit import matched_generation


def fixture():
    fitted = {"plan_sha256": "fitted", "weight_mode": "fitted", "sources": ["a", "b", "c"],
              "training_plan_sha256": "training", "checkpoint_receipt_sha256": "checkpoint",
              "implementation": {"decoder": "same"}}
    base = {**fitted, "plan_sha256": "base", "weight_mode": "base"}
    rows = [{"source_sha256": source, "procedure_equivalent": correct,
             "observed_answer_correct": correct, "meaning": {"status": status}}
            for source, correct, status in (("a", True, "equivalent"), ("b", False, "different"),
                                             ("c", False, "unknown"))]
    left = {"plan_sha256": "fitted", "weight_mode": "fitted", "training_plan_sha256": "training",
            "checkpoint_receipt_sha256": "checkpoint", "current_implementation_drift": [],
            "artifacts_verified": True, "meaning_audit": {"comparisons": rows}}
    right = {**deepcopy(left), "plan_sha256": "base", "weight_mode": "base"}
    right["meaning_audit"]["comparisons"][0].update(procedure_equivalent=False, observed_answer_correct=False)
    right["meaning_audit"]["comparisons"][0]["meaning"]["status"] = "different"
    right["meaning_audit"]["comparisons"][1]["meaning"]["status"] = "equivalent"
    return fitted, base, left, right


def test_procedure_gain_cannot_cancel_independently_witnessed_meaning_regression():
    result = matched_generation(*fixture())
    assert result["procedure"]["gains"] == 1
    assert result["new_proven_meanings"] == result["lost_proven_meanings"] == 1
    assert result["base_proven_fitted_witnessed_different"] == 1
    assert result["source_outcomes"][-1]["fitted_meaning"] == "unknown"
    assert result["general_transfer_proven"] is False


@pytest.mark.parametrize("defect", ["implementation", "population", "duplicate", "plan", "drift", "mode", "unknown_bool"])
def test_unmatched_or_incomplete_arms_cannot_produce_a_gain_claim(defect):
    fitted, base, left, right = fixture()
    if defect == "implementation":
        base["implementation"] = {"decoder": "other"}
    elif defect == "population":
        right["meaning_audit"]["comparisons"].pop()
    elif defect == "duplicate":
        right["meaning_audit"]["comparisons"][-1]["source_sha256"] = "a"
    elif defect == "plan":
        right["plan_sha256"] = "other"
    elif defect == "drift":
        right["current_implementation_drift"] = ["decoder"]
    elif defect == "mode":
        right["weight_mode"] = "fitted"
    else:
        left["meaning_audit"]["comparisons"][-1]["procedure_equivalent"] = None
    with pytest.raises(ValueError):
        matched_generation(fitted, base, left, right)
