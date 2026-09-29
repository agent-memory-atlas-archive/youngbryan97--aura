"""Later micro failures cannot erase passes or skip missing acceptance evidence."""

import hashlib
from copy import deepcopy

import pytest

from tools.adjudicate_semantic_native_micro_stages import (
    cohort_passed,
    stage_progress,
    verified_native_candidate,
    verified_reference_protocol,
)


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


def test_reference_accepts_its_declared_long_bound_but_refuses_unbounded_work():
    from tools.evaluate_semantic_native_grammar import grammar_examples

    training = {"plan_sha256": "f" * 64}
    checkpoint = {"receipt_sha256": "c" * 64}
    seed = int(training["plan_sha256"][:8], 16)
    sources = [hashlib.sha256(item.source_text.encode()).hexdigest() for item in
               grammar_examples(dataset="natural_request", seed=seed, count=3)]
    plan = {"schema": "aura.semantic_native_grammar_plan.v13",
            "dataset": "natural_request", "weight_mode": "residual", "seed": seed,
            "source_evidence": "source_text", "search_completions": 4,
            "search_nodes": 256, "max_steps": 8, "max_seconds": 14400.,
            "search_score_mode": "native_nonpositive", "sources": sources,
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
            "residual_calibration": {"report_receipt_sha256": "r"}}
    residual = {"report_receipt_sha256": "r"}
    verified_reference_protocol(plan, training, checkpoint, residual)
    for bound in (0, 14401, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="finite frozen runtime bound"):
            verified_reference_protocol({**plan, "max_seconds": bound},
                                        training, checkpoint, residual)
    with pytest.raises(ValueError, match="frozen protocol"):
        verified_reference_protocol({**plan, "search_nodes": 16}, training, checkpoint, residual)


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


@pytest.mark.parametrize("defect", ["selected_zero", "wrong_schema", "zero_scale", "drift"])
def test_residual_candidate_requires_v7_source_evidence_before_decode(monkeypatch, tmp_path, defect):
    training = {"schema": "aura.semantic_native_fit_plan.v7"}
    selected = {"step": 0}
    residual = {"selected_scale": 0.125, "candidate_step": 101,
                "current_implementation_drift": [], "report_receipt_sha256": "calibration"}
    if defect == "selected_zero":
        selected["step"] = 101
    elif defect == "wrong_schema":
        training["schema"] = "aura.semantic_native_fit_plan.v6"
    elif defect == "zero_scale":
        residual["selected_scale"] = 0.
    else:
        residual["current_implementation_drift"] = ["changed"]
    monkeypatch.setattr("tools.adjudicate_semantic_native_micro_stages.verified_native_fit",
                        lambda *args: (training, selected, {"receipt_sha256": "fit"}))
    monkeypatch.setattr("tools.verify_semantic_native_residual.verify_residual",
                        lambda *args: residual)
    checkpoint = {"step": 101, "receipt_sha256": "candidate"}
    monkeypatch.setattr("tools.evaluate_semantic_native_checkpoint.verified_document",
                        lambda path: checkpoint)
    if defect:
        with pytest.raises(ValueError, match="source-preserving gain"):
            verified_native_candidate(tmp_path, tmp_path / "fit.json", tmp_path / "calibration")


def test_positive_residual_candidate_resolves_the_measured_checkpoint(monkeypatch, tmp_path):
    training = {"schema": "aura.semantic_native_fit_plan.v7"}
    selected = {"step": 0}
    fit = {"receipt_sha256": "fit"}
    residual = {"selected_scale": 0.125, "candidate_step": 101,
                "current_implementation_drift": [], "report_receipt_sha256": "calibration"}
    monkeypatch.setattr("tools.adjudicate_semantic_native_micro_stages.verified_native_fit",
                        lambda *args: (training, selected, fit))
    monkeypatch.setattr("tools.verify_semantic_native_residual.verify_residual",
                        lambda *args: residual)
    paths = []
    def load(path):
        paths.append(path)
        return {"step": 101, "receipt_sha256": "candidate"}
    monkeypatch.setattr("tools.evaluate_semantic_native_checkpoint.verified_document", load)
    outcome = verified_native_candidate(tmp_path, tmp_path / "fit.json", tmp_path / "calibration")
    assert outcome[1]["receipt_sha256"] == "candidate"
    assert outcome[3] == residual
    assert paths == [tmp_path / "checkpoint-101.json"]
