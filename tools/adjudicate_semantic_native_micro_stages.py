#!/usr/bin/env python3
"""Advance verified native micro stages without re-decoding a passed cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cohort_passed(comparison, *, population):
    measured = comparison["comparison"]
    intervention = comparison["source_intervention"]
    if (measured["population"] != population or intervention["population"] != population
            or len(intervention["source_outcomes"]) != population):
        raise ValueError("native micro stage population differs from its protocol")
    return (measured["procedure"]["fitted_correct"] == population
            and measured["answer"]["fitted_correct"] == population
            and measured["procedure"]["regressions"] == 0
            and measured["answer"]["regressions"] == 0
            and all(row["arms"]["fitted"]["decode_status"] == "completed"
                    and row["arms"]["fitted"]["bound_forced_completion"] is False
                    for row in intervention["source_outcomes"]))


def stage_progress(reference, *, controls=None, retained=None):
    """A failed later stage preserves earlier passes and grants no broad authority."""
    reference_ok = cohort_passed(reference, population=3)
    if not reference_ok and (controls is not None or retained is not None):
        raise ValueError("native micro stages cannot skip a failed reference stage")
    controls_ok = None
    if controls is not None:
        if (controls["fitted_report_receipt_sha256"] != reference["fitted_report_receipt_sha256"]
                or controls["base_report_receipt_sha256"] != reference["base_report_receipt_sha256"]
                or controls["erasure_report_receipt_sha256"] != reference["erasure_report_receipt_sha256"]
                or controls["comparison"] != reference["comparison"]
                or controls["source_intervention"] != reference["source_intervention"]):
            raise ValueError("native micro reference stage was replaced")
        relation = controls["relation_transfer"]
        if (relation["reference_population"] != 3 or relation["controlled_population"] != 9
                or type(relation["mechanism_micro_probe_passed"]) is not bool
                or relation["relations_recovered"] != sum(
                    row["expected_relation_recovered"] is True for row in relation["source_outcomes"])
                or len(relation["source_outcomes"]) != 9
                or relation["mechanism_micro_probe_passed"] != (relation["relations_recovered"] == 9)
                or relation["reference_stage_reused_without_redecode"] is not True):
            raise ValueError("native micro relation stage lacks its complete frozen controls")
        controls_ok = relation["mechanism_micro_probe_passed"]
    if retained is not None and controls_ok is not True:
        raise ValueError("native micro stages cannot skip the relation stage")
    retained_ok = None if retained is None else cohort_passed(retained, population=6)
    artifacts = (reference, *(item for item in (controls, retained) if item is not None))
    if any((item["training_plan_sha256"], item["checkpoint_receipt_sha256"])
           != (reference["training_plan_sha256"], reference["checkpoint_receipt_sha256"])
           for item in artifacts):
        raise ValueError("native micro stages changed the selected candidate")
    cohorts = (reference, *((retained,) if retained is not None else ()))
    dependent = sum(item["source_intervention"]["source_dependent_gains"] for item in cohorts)
    stage_values = (("reference_requests", reference_ok), ("relation_controls", controls_ok),
                    ("retained_requests", retained_ok))
    pending = next((name for name, passed in stage_values if passed is not True), None)
    ready = pending is None and dependent > 0
    return {"stages": [{"stage": name, "status": "not_run" if passed is None
                        else "passed" if passed else "failed"} for name, passed in stage_values],
            "current_stage": pending or ("full_development" if ready else "source_attribution"),
            "full_development_ready": ready, "source_dependent_gains": dependent,
            "passed_stages_redecoded": False, "general_transfer_proven": False,
            "broad_gain_proven": False, "serving_authority": False}


def verified_native_fit(training_directory, fit_verification):
    """Require the complete fit, selected weights, and unchanged replay evidence."""
    from tools.evaluate_semantic_native_checkpoint import (
        selected_checkpoint,
        verified_document,
    )
    training, checkpoint = selected_checkpoint(training_directory)
    training_report = verified_document(training_directory / "report.json")
    fit = verified_document(fit_verification)
    if (fit.get("artifacts_verified") is not True
            or fit.get("current_implementation_drift") != []
            or fit.get("training_plan_sha256") != training["plan_sha256"]
            or fit.get("training_receipt_sha256") != training_report["receipt_sha256"]
            or fit.get("selected_checkpoint_receipt_sha256") != checkpoint["receipt_sha256"]
            or any(hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != sha
                   for path, sha in training["implementation"].items())):
        raise ValueError("native micro stages lack unchanged verified fitting evidence")
    return training, checkpoint, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", required=True, type=Path)
    parser.add_argument("--fit-verification", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--controls-directory", type=Path)
    parser.add_argument("--retained-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from tools.compare_semantic_native_grammar_fit import compare_directories
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    training, checkpoint, fit = verified_native_fit(args.training_directory, args.fit_verification)

    def compare(root, controls=None):
        return compare_directories(fitted_directory=root / "fitted",
            base_directory=root / "base", erasure_directory=root / "erasure",
            training_directory=args.training_directory, relation_controls_directory=controls)

    reference_plan = verified_document(args.reference_root / "fitted/plan.json", "plan_sha256")
    from tools.evaluate_semantic_native_grammar import grammar_examples
    expected_sources = [hashlib.sha256(item.source_text.encode()).hexdigest() for item in
                        grammar_examples(dataset="natural_request", seed=reference_plan["seed"], count=3)]
    expected = {"dataset": "natural_request", "weight_mode": "fitted",
                "seed": int(training["plan_sha256"][:8], 16),
                "source_evidence": "source_text", "search_completions": 4,
                "search_nodes": 256, "max_steps": 8, "max_seconds": 3600.,
                "search_score_mode": "native_nonpositive", "sources": expected_sources}
    if (any(reference_plan.get(key) != value for key, value in expected.items())
            or reference_plan.get("prefix_strategy", "full") != "full"
            or reference_plan["training_plan_sha256"] != training["plan_sha256"]
            or reference_plan["checkpoint_receipt_sha256"] != checkpoint["receipt_sha256"]):
        raise ValueError("native micro reference stage changed its frozen protocol")
    reference = compare(args.reference_root)
    controls = None if args.controls_directory is None else compare(args.reference_root, args.controls_directory)
    retained = None if args.retained_root is None else compare(args.retained_root)
    if args.retained_root is not None:
        retained_plan = verified_document(args.retained_root / "fitted/plan.json", "plan_sha256")
        excluded = {"schema", "plan_sha256", "dataset", "seed", "sources", "implementation",
                    "source_cohort_basis"}
        additions = {"tools/semantic_native_retained_sources.py",
                     "core/learning/semantic_program_feature_materialization.py"}
        if (reference_plan["dataset"] != "natural_request"
                or retained_plan["dataset"] != "retained_validation"
                or {key: value for key, value in reference_plan.items() if key not in excluded}
                    != {key: value for key, value in retained_plan.items() if key not in excluded}
                or set(retained_plan["implementation"]) - set(reference_plan["implementation"]) != additions
                or any(retained_plan["implementation"].get(path) != sha
                       for path, sha in reference_plan["implementation"].items())):
            raise ValueError("native micro retained stage changed the common protocol")
    body = {"schema": "aura.native_micro_stage_progress.v1",
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
            "fit_verification_receipt_sha256": fit["receipt_sha256"],
            "comparison_receipts": {"reference": reference["receipt_sha256"],
                "controls": None if controls is None else controls["receipt_sha256"],
                "retained": None if retained is None else retained["receipt_sha256"]},
            **stage_progress(reference, controls=controls, retained=retained)}
    result = {**body, "receipt_sha256": digest(body)}
    _save_if_absent(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
