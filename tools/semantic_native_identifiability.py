"""Check exact-path scoring contradictions before native model allocation."""

from __future__ import annotations

from pathlib import Path

IDENTIFIABILITY_CONTRACT = {
    "schema": "aura.native_identifiability_preflight_contract.v1",
    "basis": "ordered_causal_token_choices_and_scored_target_positions",
    "required": "no_conflicting_exact_targets_on_identical_scoring_inputs",
    "partitions": "captured_fit_and_source_calibration",
    "held_labels_used": False,
    "learnability_proven": False,
}


def identifiability_preflight(plan: dict, supervision: dict) -> dict:
    from tools.probe_semantic_proposer_crossfit import _digest
    from tools.verify_semantic_native_fit import audit_grammar_identifiability

    if (plan.get("objective") not in {"grammar_choices", "grammar_source_pairs"}
            or plan.get("plan_sha256") != _digest({key: value for key, value in plan.items()
                                                   if key != "plan_sha256"})
            or supervision.get("plan_sha256") != plan["plan_sha256"]
            or supervision.get("receipt_sha256") != _digest({key: value for key, value in
                supervision.items() if key != "receipt_sha256"})):
        raise ValueError("native identifiability preflight needs bound grammar supervision")
    fitting, calibration = (plan.get(name) for name in ("captured_fit_ids", "calibration_ids"))
    if (any(not isinstance(ids, (list, tuple)) or not ids
            or any(not isinstance(identity, str) or not identity for identity in ids)
            or len(ids) != len(set(ids))
            for ids in (fitting, calibration)) or set(fitting) & set(calibration)
            or (set(fitting) | set(calibration)) & set(plan.get("held_ids", ()))):
        raise ValueError("native identifiability source partitions differ")
    rows = supervision.get("rows")
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) or not isinstance(row.get("source"), str) for row in rows)
            or {row.get("source") for row in rows} != set(fitting) | set(calibration)):
        raise ValueError("native identifiability supervision is incomplete or crosses a partition")
    audits = {name: audit_grammar_identifiability(
        [row for row in rows if row["source"] in identities]) for name, identities in (
            ("fit", set(fitting)), ("calibration", set(calibration)),
            ("combined", set(fitting) | set(calibration)))}
    requirement = plan.get("grammar_identifiability_contract")
    if requirement is not None and requirement != IDENTIFIABILITY_CONTRACT:
        raise ValueError("native identifiability requirement differs")
    body = {"schema": "aura.native_identifiability_preflight.v1",
            "plan_sha256": plan["plan_sha256"],
            "supervision_receipt_sha256": supervision["receipt_sha256"],
            "audits": audits, "strict_requirement": requirement is not None,
            "exact_targets_identifiable": audits["combined"]["contradictory_scoring_input_groups"] == 0,
            "model_weights_loaded": False, "held_labels_used": False,
            "learnability_proven": False, "semantic_transfer_proven": False,
            "qualification_evidence": False, "serving_authority": False}
    return {**body, "receipt_sha256": _digest(body)}


def save_identifiability_preflight(directory: Path, plan: dict, supervision: dict) -> dict:
    from tools.probe_semantic_proposer_crossfit import _save_if_absent

    receipt = identifiability_preflight(plan, supervision)
    _save_if_absent(directory / "identifiability-preflight.json", receipt)
    if receipt["strict_requirement"] and not receipt["exact_targets_identifiable"]:
        raise ValueError("native exact-path requirement has contradictory causal scoring inputs")
    return receipt


def verify_identifiability_preflight(directory: Path, plan: dict, supervision: dict) -> dict:
    from tools.evaluate_semantic_native_checkpoint import verified_document

    expected = identifiability_preflight(plan, supervision)
    observed = verified_document(directory / "identifiability-preflight.json")
    if (observed != expected
            or expected["strict_requirement"] and not expected["exact_targets_identifiable"]):
        raise ValueError("native fit identifiability preflight differs or cannot meet its requirement")
    return expected
