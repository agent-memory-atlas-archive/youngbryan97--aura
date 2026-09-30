"""Impossible exact-path supervision stops before a model can be loaded."""

import json

import pytest

from tests.test_semantic_native_identifiability import decision
from tools.probe_semantic_proposer_crossfit import _digest
from tools.semantic_native_identifiability import (
    IDENTIFIABILITY_CONTRACT,
    identifiability_preflight,
    save_identifiability_preflight,
    verify_identifiability_preflight,
)


def bound(rows, *, strict=True):
    plan = {"objective": "grammar_source_pairs", "captured_fit_ids": ["a"],
            "calibration_ids": ["b"], "held_ids": ["held"]}
    if strict:
        plan["grammar_identifiability_contract"] = dict(IDENTIFIABILITY_CONTRACT)
    plan["plan_sha256"] = _digest(plan)
    supervision = {"plan_sha256": plan["plan_sha256"], "rows": rows}
    supervision["receipt_sha256"] = _digest(supervision)
    return plan, supervision


def test_compatible_duplicates_pass_without_claiming_learnability(tmp_path):
    plan, supervision = bound(decision() + decision("b"))
    receipt = save_identifiability_preflight(tmp_path, plan, supervision)
    assert receipt["exact_targets_identifiable"] is True
    assert receipt["audits"]["combined"]["maximum_exact_teacher_decisions"] == 2
    assert receipt["audits"]["combined"]["duplicate_scoring_input_groups"] == 1
    assert all(receipt[name] is False for name in (
        "model_weights_loaded", "learnability_proven", "semantic_transfer_proven",
        "qualification_evidence", "serving_authority", "held_labels_used"))
    assert json.loads((tmp_path / "identifiability-preflight.json").read_bytes()) == receipt
    assert verify_identifiability_preflight(tmp_path, plan, supervision) == receipt


def test_cross_partition_contradiction_is_retained_then_rejected(tmp_path):
    plan, supervision = bound(decision() + decision("b", 1))
    with pytest.raises(ValueError, match="contradictory"):
        save_identifiability_preflight(tmp_path, plan, supervision)
    receipt = json.loads((tmp_path / "identifiability-preflight.json").read_bytes())
    assert receipt["exact_targets_identifiable"] is False
    assert receipt["audits"]["fit"]["contradictory_scoring_input_groups"] == 0
    assert receipt["audits"]["calibration"]["contradictory_scoring_input_groups"] == 0
    assert receipt["audits"]["combined"]["maximum_exact_teacher_decisions"] == 1
    assert receipt["audits"]["combined"]["contradictory_scoring_input_groups"] == 1


def test_probabilistic_tasks_can_measure_conflicts_without_strict_rejection(tmp_path):
    plan, supervision = bound(decision() + decision("b", 1), strict=False)
    receipt = save_identifiability_preflight(tmp_path, plan, supervision)
    assert receipt["strict_requirement"] is False
    assert receipt["exact_targets_identifiable"] is False


def test_in_memory_tuple_partitions_match_their_json_form():
    plan, supervision = bound(decision() + decision("b"))
    for name in ("captured_fit_ids", "calibration_ids", "held_ids"):
        plan[name] = tuple(plan[name])
    assert identifiability_preflight(plan, supervision) == identifiability_preflight(
        json.loads(json.dumps(plan)), supervision)


@pytest.mark.parametrize("fault", ["missing", "self_signed_false_pass", "bound_other_plan"])
def test_independent_verification_requires_the_actual_preflight(tmp_path, fault):
    plan, supervision = bound(decision() + decision("b"))
    receipt = save_identifiability_preflight(tmp_path, plan, supervision)
    path = tmp_path / "identifiability-preflight.json"
    path.unlink()
    if fault != "missing":
        receipt["exact_targets_identifiable"] = False
        if fault == "bound_other_plan":
            receipt["plan_sha256"] = "f" * 64
        receipt["receipt_sha256"] = _digest({key: value for key, value in receipt.items()
                                              if key != "receipt_sha256"})
        path.write_text(json.dumps(receipt))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_identifiability_preflight(tmp_path, plan, supervision)


@pytest.mark.parametrize("fault", ["plan_digest", "supervision_digest", "other_plan", "held",
                                    "missing", "overlap", "repeated", "requirement", "objective"])
def test_preflight_cannot_use_unbound_or_held_evidence(fault):
    plan, supervision = bound(decision() + decision("b"))
    if fault == "plan_digest":
        plan["calibration_ids"] = ["other"]
    elif fault == "supervision_digest":
        supervision["rows"][0]["tokens"][0] = 13
    elif fault == "other_plan":
        supervision["plan_sha256"] = "f" * 64
    elif fault == "held":
        supervision["rows"].extend(decision("held"))
    elif fault == "missing":
        supervision["rows"] = decision()
    elif fault == "overlap":
        plan["calibration_ids"] = ["a"]
    elif fault == "repeated":
        plan["captured_fit_ids"] = ["a", "a"]
    elif fault == "requirement":
        plan["grammar_identifiability_contract"]["held_labels_used"] = True
    else:
        plan["objective"] = "token"
    if fault not in {"plan_digest", "supervision_digest", "other_plan"}:
        plan["plan_sha256"] = _digest({key: value for key, value in plan.items() if key != "plan_sha256"})
        supervision["plan_sha256"] = plan["plan_sha256"]
        supervision["receipt_sha256"] = _digest({key: value for key, value in supervision.items()
                                                 if key != "receipt_sha256"})
    with pytest.raises(ValueError):
        identifiability_preflight(plan, supervision)
