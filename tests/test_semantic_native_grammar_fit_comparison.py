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


def residual_fixture():
    candidate, base, left, right = fixture()
    candidate.update(weight_mode="residual", schema="aura.semantic_native_grammar_plan.v9",
                     checkpoint_receipt_sha256="candidate", residual_calibration={
                         "baseline_checkpoint_receipt_sha256": "checkpoint",
                         "source_only": True, "serving_authority": False})
    candidate["implementation"] = {**candidate["implementation"],
        "tools/calibrate_semantic_native_residual.py": "calibration",
        "tools/verify_semantic_native_residual.py": "verification"}
    base["schema"] = "aura.semantic_native_grammar_plan.v8"
    left.update(weight_mode="residual", checkpoint_receipt_sha256="candidate")
    return candidate, base, left, right


def test_residual_matches_only_its_bound_baseline_and_common_implementation():
    assert matched_generation(*residual_fixture())["candidate_weight_mode"] == "residual"


@pytest.mark.parametrize("defect", ["checkpoint", "source_only", "authority", "schema", "decoder", "extra"])
def test_residual_cannot_excuse_protocol_or_baseline_drift(defect):
    candidate, base, left, right = residual_fixture()
    if defect == "checkpoint":
        candidate["residual_calibration"]["baseline_checkpoint_receipt_sha256"] = "other"
    elif defect == "source_only":
        candidate["residual_calibration"]["source_only"] = False
    elif defect == "authority":
        candidate["residual_calibration"]["serving_authority"] = True
    elif defect == "schema":
        base["schema"] = "aura.semantic_native_grammar_plan.v7"
    elif defect == "decoder":
        candidate["implementation"]["decoder"] = "changed"
    else:
        candidate["implementation"]["other"] = "changed"
    with pytest.raises(ValueError):
        matched_generation(candidate, base, left, right)


def test_factorized_comparison_requires_exact_added_implementations_and_lineage():
    candidate, base, left, right = residual_fixture()
    candidate.update(weight_mode="factorized", schema="aura.semantic_native_grammar_plan.v10")
    candidate["factorized_residual"] = candidate.pop("residual_calibration")
    candidate["implementation"].update({
        "core/learning/semantic_native_factorized_residual.py": "factor",
        "tools/factor_semantic_native_residual.py": "build",
        "tools/verify_semantic_native_factorized_residual.py": "verify"})
    left["weight_mode"] = "factorized"
    assert matched_generation(candidate, base, left, right)["candidate_weight_mode"] == "factorized"
    candidate["implementation"].pop("core/learning/semantic_native_factorized_residual.py")
    with pytest.raises(ValueError, match="lineage"):
        matched_generation(candidate, base, left, right)
