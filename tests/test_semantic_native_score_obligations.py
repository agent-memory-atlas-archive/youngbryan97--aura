"""A calculated score repair is a finite constraint, not a generalization claim."""

import hashlib
import json
import math

import pytest

from tools.audit_semantic_native_score_obligations import (
    audit_directory,
    digest,
    ranking_obligation,
    source_obligations,
)


def test_score_correction_solves_the_declared_margin_exactly():
    row = ranking_obligation([-7., -3., -4.], 0, margin=.5)
    assert row["strongest_rival_index"] == 1
    assert row["relative_score_correction_needed"] == 4.5
    corrected = [-2.5, -3., -4.]
    assert ranking_obligation(corrected, 0, margin=.5)["strict_margin_satisfied"]


def test_a_tie_can_pass_greedy_tie_break_without_satisfying_a_strict_margin():
    row = {"source": "source", "decisions": [
        {"kind": "termination", "scores": [-2., -2.], "correct_index": 0}]}
    result = source_obligations(row, margin=.1)
    assert result["teacher_path_exact"] is True
    assert result["greedy_strict_margin_satisfied"] is False
    assert result["decisions"][0]["relative_score_correction_needed"] == .1


def test_local_constraints_cannot_certify_unmeasured_global_graphs():
    row = {"source": "source", "decisions": [
        {"kind": "operation", "scores": [-1., -2.], "correct_index": 0},
        {"kind": "termination", "scores": [-1., -2.], "correct_index": 0}],
        "whole_graph": {"scores": [-4., -3.], "positive_index": 0,
                        "program_sha256s": ["correct", "rival"]}}
    result = source_obligations(row, margin=.1)
    assert result["greedy_strict_margin_satisfied"] is True
    assert result["whole_graph"]["relative_score_correction_needed"] == 1.1
    assert result["global_search_certified"] is False
    assert result["unseen_correctness_certified"] is False


def test_singleton_has_no_ranking_obligation_but_still_needs_a_positive_declared_margin():
    result = ranking_obligation([-2.], 0, margin=.1)
    assert result["observed_margin"] is None and result["strict_margin_satisfied"]
    assert result["relative_score_correction_needed"] == 0.


@pytest.mark.parametrize("margin", [0, -.1, True, math.inf, math.nan])
def test_invalid_margin_is_not_a_zero_work_certificate(margin):
    with pytest.raises(ValueError):
        ranking_obligation([0., -1.], 0, margin=margin)


@pytest.mark.parametrize("scores,correct", [([], 0), ([True, -1.], 0), ([math.nan, -1.], 0),
                                           ([0., -1.], True), ([0., -1.], 2)])
def test_missing_or_invalid_measurements_are_refused(scores, correct):
    with pytest.raises(ValueError):
        ranking_obligation(scores, correct, margin=.1)


def test_duplicate_graph_identity_cannot_supply_an_easy_negative():
    row = {"source": "source", "decisions": [{"kind": "termination", "scores": [0.], "correct_index": 0}],
           "whole_graph": {"scores": [0., -3.], "positive_index": 0,
                           "program_sha256s": ["correct", "correct"]}}
    with pytest.raises(ValueError, match="distinct measured"):
        source_obligations(row, margin=.1)


@pytest.mark.parametrize("defect", [None, "weights", "calibration", "held_fit", "alternatives"])
def test_audit_binds_real_files_and_rejects_changed_custody(tmp_path, monkeypatch, defect):
    from tools import audit_semantic_native_path_calibration as calibration

    def save(name, body, key="receipt_sha256"):
        payload = {**body, key: digest(body)}
        (tmp_path / name).write_text(json.dumps(payload))
        return payload

    training = save("plan.json", {"held_labels_used_for_fit_or_selection": defect == "held_fit"}, "plan_sha256")
    supervision = save("supervision.json", {"graph_rows": []})
    measured = save("calibration-paths-0.json", {"plan_sha256": training["plan_sha256"], "step": 0,
        "rows": [{"source": "source", "decisions": [{"kind": "reference", "correct_index": 0,
                 "scores": [-2., -1.], "choices": ["input:0", "input:1"]},
                 {"kind": "termination", "correct_index": 0, "scores": [0.], "choices": ["finish"]}]}]})
    weights = b"opaque checkpoint whose bytes are audited without loading arrays"
    (tmp_path / "checkpoint-0.safetensors").write_bytes(weights)
    checkpoint = save("checkpoint-0.json", {"step": 0, "plan_sha256": training["plan_sha256"],
        "weights_sha256": hashlib.sha256(weights).hexdigest(),
        "calibration_path_receipt_sha256": measured["receipt_sha256"]})
    save("report.json", {"plan_sha256": training["plan_sha256"], "checkpoints": [checkpoint],
                        "supervision_receipt_sha256": supervision["receipt_sha256"]})
    choices = ["input:0", "input:1"] if defect != "alternatives" else ["input:1", "input:0"]
    monkeypatch.setattr(calibration, "calibration_competitions", lambda *_a: {"source": [
        [{"kind": "reference", "correct_index": 0, "choice": choice} for choice in choices],
        [{"kind": "termination", "correct_index": 0, "choice": "finish"}]]})
    if defect == "weights":
        (tmp_path / "checkpoint-0.safetensors").write_bytes(b"changed")
    elif defect == "calibration":
        measured["rows"][0]["decisions"][0]["scores"][0] = 0.
        (tmp_path / "calibration-paths-0.json").write_text(json.dumps(measured))
    if defect is not None:
        with pytest.raises(ValueError):
            audit_directory(tmp_path, margin=.1)
    else:
        result = audit_directory(tmp_path, margin=.1)
        assert result["checkpoints"][0]["decision_deficits_by_kind"] == {"reference": 1}
        assert result["current_training_contract_replayed"] is False
        assert result["fit_qualification_evidence"] is False
