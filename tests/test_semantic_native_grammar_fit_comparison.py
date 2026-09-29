"""Generated fit claims require identical source/scoring protocols and honest unknowns."""

from copy import deepcopy

import pytest

from tools.compare_semantic_native_grammar_fit import (
    matched_generation,
    matched_source_intervention,
)


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


def test_cached_grouped_residual_matches_only_its_identical_execution_baseline():
    candidate, base, left, right = residual_fixture()
    candidate["schema"] = base["schema"] = "aura.semantic_native_grammar_plan.v15"
    candidate["prefix_strategy"] = base["prefix_strategy"] = "trie"
    candidate["decision_score_execution"] = base["decision_score_execution"] = "causal_groups"
    candidate["implementation"]["core/learning/frozen_prefix_branches.py"] = "cached"
    base["implementation"]["core/learning/frozen_prefix_branches.py"] = "cached"
    assert matched_generation(candidate, base, left, right)["candidate_weight_mode"] == "residual"
    base["prefix_strategy"] = "full"
    with pytest.raises(ValueError, match="differ beyond fitted weights"):
        matched_generation(candidate, base, left, right)


@pytest.mark.parametrize("dataset,base_schema,window", [
    ("natural_request", "v3", False),
    ("relation_transfer_controls", "v11", False),
    ("retained_validation", "v6", False),
    ("retained_validation", "v12", True),
])
def test_joint_residual_matches_the_same_full_prefix_cohort(dataset, base_schema, window):
    candidate, base, left, right = residual_fixture()
    candidate["schema"] = "aura.semantic_native_grammar_plan.v13"
    candidate["dataset"] = base["dataset"] = dataset
    base["schema"] = f"aura.semantic_native_grammar_plan.{base_schema}"
    if window:
        candidate["source_window"] = base["source_window"] = {"offset": 0, "count": 3}
    assert matched_generation(candidate, base, left, right)["candidate_weight_mode"] == "residual"
    base["schema"] = "aura.semantic_native_grammar_plan.v8"
    with pytest.raises(ValueError, match="lineage"):
        matched_generation(candidate, base, left, right)


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


def intervention_fixture():
    fitted, base, left, right = fixture()
    fitted.update(source_evidence="source_text", search_completions=4)
    base.update(source_evidence="source_text", search_completions=4)
    erasure = {**fitted, "plan_sha256": "erasure",
               "source_evidence": "source_token_erasure"}
    erasure_verified = deepcopy(left)
    erasure_verified["plan_sha256"] = "erasure"
    erasure_verified["meaning_audit"]["comparisons"][0].update(
        procedure_equivalent=False, observed_answer_correct=False)

    def row(source, program):
        return {"source_sha256": source, "program": program,
                "decode_status": "completed", "bound_forced_completion": False,
                "search": {"proposals": [{"program": program}],
                           "observed_program_reach": True,
                           "requested_top_k_proven": False,
                           "halt_reason": "node_bound"}}

    fitted_rows = [row(source, {"id": f"fitted-{source}"}) for source in fitted["sources"]]
    base_rows = [row(source, {"id": f"base-{source}"}) for source in fitted["sources"]]
    erasure_rows = [row(source, {"id": f"erasure-{source}"}) for source in fitted["sources"]]
    return (fitted, base, erasure, left, right, erasure_verified,
            fitted_rows, base_rows, erasure_rows)


def test_micro_probe_requires_exact_source_dependent_gain_and_reports_weak_control():
    arms = intervention_fixture()
    result = matched_source_intervention(*arms)
    assert result["bounded_micro_probe_passed"] is True
    assert result["exact_gains"] == result["source_dependent_gains"] == 1
    assert result["exact_regressions"] == 0
    assert result["baseline_regression_control_informative"] is False
    assert result["source_outcomes"][0]["arms"]["fitted"]["requested_top_k_proven"] is False
    assert result["general_transfer_proven"] is False

    *prefix, fitted_rows, base_rows, erasure_rows = intervention_fixture()
    prefix[5]["meaning_audit"]["comparisons"][0].update(
        procedure_equivalent=True, observed_answer_correct=True)
    erasure_rows[0]["program"] = fitted_rows[0]["program"]
    result = matched_source_intervention(*prefix, fitted_rows, base_rows, erasure_rows)
    assert result["bounded_micro_probe_passed"] is False
    assert result["source_dependent_gains"] == 0


def test_cached_grouped_source_erasure_keeps_three_arms_matched():
    arms = list(intervention_fixture())
    for plan in arms[:3]:
        plan.update(schema="aura.semantic_native_grammar_plan.v15",
                    prefix_strategy="trie", decision_score_execution="causal_groups")
    assert matched_source_intervention(*arms)["source_dependent_gains"] == 1
    arms[2]["decision_score_execution"] = "individual"
    with pytest.raises(ValueError, match="beyond erasure"):
        matched_source_intervention(*arms)


@pytest.mark.parametrize("defect", ["erasure_mode", "source_population", "erasure_proof",
                                    "erasure_row", "missing_search", "regression"])
def test_micro_probe_refuses_unmatched_arms_or_flags_regressions(defect):
    arms = list(intervention_fixture())
    if defect == "erasure_mode":
        arms[2]["source_evidence"] = "source_text"
    elif defect == "source_population":
        arms[2]["sources"] = ["a", "b", "other"]
    elif defect == "erasure_proof":
        arms[5]["artifacts_verified"] = False
    elif defect == "erasure_row":
        arms[8][0]["source_sha256"] = "other"
    elif defect == "missing_search":
        arms[6][0]["search"] = None
    else:
        arms[4]["meaning_audit"]["comparisons"][1]["procedure_equivalent"] = True
        result = matched_source_intervention(*arms)
        assert result["exact_regressions"] == 1
        assert result["bounded_micro_probe_passed"] is False
        return
    with pytest.raises(ValueError):
        matched_source_intervention(*arms)
