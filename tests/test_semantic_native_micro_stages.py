"""Later micro failures cannot erase passes or skip missing acceptance evidence."""

from copy import deepcopy

import pytest

from tools.adjudicate_semantic_native_micro_stages import cohort_passed, stage_progress


def cohort(population, *, dependent=0):
    metric = {"fitted_correct": population, "regressions": 0}
    return {"training_plan_sha256": "training", "checkpoint_receipt_sha256": "weights",
        "fitted_report_receipt_sha256": "fitted", "base_report_receipt_sha256": "base",
        "erasure_report_receipt_sha256": "erasure", "comparison": {
            "population": population, "procedure": dict(metric), "answer": dict(metric)},
        "source_intervention": {"population": population, "source_dependent_gains": dependent,
            "source_outcomes": [{"arms": {"fitted": {
                "decode_status": "completed", "bound_forced_completion": False}}}
                for _ in range(population)]}}


def controls(reference, *, recovered=9):
    return {**deepcopy(reference), "relation_transfer": {
        "reference_population": 3, "controlled_population": 9,
        "relations_recovered": recovered, "mechanism_micro_probe_passed": recovered == 9,
        "source_outcomes": [{"expected_relation_recovered": ordinal < recovered}
                            for ordinal in range(9)],
        "reference_stage_reused_without_redecode": True}}


def test_passed_reference_advances_without_unrun_controls_becoming_passes():
    result = stage_progress(cohort(3))
    assert result["stages"] == [
        {"stage": "reference_requests", "status": "passed"},
        {"stage": "relation_controls", "status": "not_run"},
        {"stage": "retained_requests", "status": "not_run"}]
    assert result["current_stage"] == "relation_controls"
    assert result["full_development_ready"] is False


def test_later_failure_keeps_the_passed_reference_stage():
    reference = cohort(3)
    result = stage_progress(reference, controls=controls(reference, recovered=8))
    assert result["stages"][0]["status"] == "passed"
    assert result["stages"][1]["status"] == "failed"
    assert result["current_stage"] == "relation_controls"
    assert result["passed_stages_redecoded"] is False


def test_all_exact_cohorts_still_require_source_dependent_gain():
    reference = cohort(3)
    result = stage_progress(reference, controls=controls(reference), retained=cohort(6))
    assert result["current_stage"] == "source_attribution"
    assert result["full_development_ready"] is False
    result = stage_progress(reference, controls=controls(reference), retained=cohort(6, dependent=1))
    assert result["current_stage"] == "full_development"
    assert result["full_development_ready"] is True
    assert result["general_transfer_proven"] is False
    assert result["serving_authority"] is False


@pytest.mark.parametrize("defect", ["partial", "answer", "regression", "forced", "disconnected"])
def test_every_cohort_obligation_must_be_measured_and_satisfied(defect):
    reference = cohort(3)
    retained = cohort(6, dependent=1)
    if defect == "partial":
        retained["source_intervention"]["source_outcomes"].pop()
        with pytest.raises(ValueError, match="population"):
            cohort_passed(retained, population=6)
        return
    if defect == "answer":
        retained["comparison"]["answer"]["fitted_correct"] = 5
    elif defect == "regression":
        retained["comparison"]["procedure"]["regressions"] = 1
    elif defect == "forced":
        retained["source_intervention"]["source_outcomes"][0]["arms"]["fitted"]["bound_forced_completion"] = True
    else:
        retained["source_intervention"]["source_outcomes"][0]["arms"]["fitted"]["decode_status"] = "search_without_completion"
    result = stage_progress(reference, controls=controls(reference), retained=retained)
    assert [row["status"] for row in result["stages"]] == ["passed", "passed", "failed"]
    assert result["current_stage"] == "retained_requests"


@pytest.mark.parametrize("defect", ["reference", "weights", "unrun", "skipped", "unknown"])
def test_a_changed_or_skipped_stage_cannot_inherit_the_passes(defect):
    reference = cohort(3)
    control = controls(reference)
    retained = cohort(6, dependent=1)
    if defect == "reference":
        control["fitted_report_receipt_sha256"] = "other"
    elif defect == "weights":
        retained["checkpoint_receipt_sha256"] = "other"
    elif defect == "unrun":
        control = None
    elif defect == "skipped":
        reference["comparison"]["answer"]["fitted_correct"] = 2
    else:
        control["relation_transfer"]["mechanism_micro_probe_passed"] = None
    with pytest.raises(ValueError):
        stage_progress(reference, controls=control, retained=retained)
