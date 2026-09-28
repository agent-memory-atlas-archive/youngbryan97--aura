#!/usr/bin/env python3
"""Independently reconstruct every source-only residual combination and selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def independent_factor_admission(measured, sources):
    from core.learning.semantic_native_path_calibration import native_path_profile, native_path_totals

    kinds = ("operation", "reference", "termination")
    for rows in measured.values():
        native_path_totals(rows, sources)
    base = measured[0.]
    exact_base = {row["source"] for row in base
                  if native_path_profile(row["decisions"])["exact_teacher_path"]}
    options = []
    for values in product(sorted(measured), repeat=3):
        scales = dict(zip(kinds, values, strict=True))
        combined = []
        for index, (source, baseline) in enumerate(zip(sources, base, strict=True)):
            decisions = []
            for ordinal, reference in enumerate(baseline["decisions"]):
                actual = measured[scales[reference["kind"]]][index]["decisions"][ordinal]
                if any(actual[key] != reference[key] for key in ("kind", "choices", "correct_index")):
                    raise ValueError("independent factorization source supervision differs")
                decisions.append(actual)
            combined.append({"source": source, "decisions": decisions})
        exact = {row["source"] for row in combined
                 if native_path_profile(row["decisions"])["exact_teacher_path"]}
        options.append({"scales": scales, "totals": native_path_totals(combined, sources),
                        "lost_baseline_sources": sorted(exact_base - exact),
                        "new_exact_sources": sorted(exact - exact_base), "eligible": exact_base <= exact})
    selected = min((row for row in options if row["eligible"]), key=lambda row: (
        -row["totals"]["exact_teacher_paths"], row["totals"]["mean_source_conditional_loss"],
        sum(row["scales"].values()), tuple(row["scales"][kind] for kind in kinds)))
    return {"selected_scales": selected["scales"], "selected_totals": selected["totals"],
            "adjudication": options}


def verify_factorized_residual(path, training_directory):
    from core.learning.semantic_native_factorized_residual import FACTORIZED_RESIDUAL_CONTRACT
    from tools.evaluate_semantic_native_checkpoint import verified_document
    from tools.verify_semantic_native_residual import verify_residual

    report = verified_document(path)
    directory = Path(report["calibration_directory"])
    verified = verify_residual(directory, training_directory)
    plan = verified_document(directory / "plan.json", "plan_sha256")
    if (report.get("schema") != "aura.native_factorized_residual.v1"
            or report.get("contract") != FACTORIZED_RESIDUAL_CONTRACT
            or report.get("calibration_plan_sha256") != plan["plan_sha256"]
            or report.get("calibration_report_receipt_sha256") != verified["report_receipt_sha256"]
            or report.get("sources") != plan["sources"]
            or set(report.get("implementation", {})) != {
                "core/learning/semantic_native_factorized_residual.py",
                "tools/factor_semantic_native_residual.py"}
            or any(report.get(key) != plan[key] for key in (
                "training_plan_sha256", "baseline_checkpoint_receipt_sha256",
                "candidate_checkpoint_receipt_sha256"))
            or any(report.get(key) is not False for key in (
                "held_labels_used", "serving_authority", "qualification_evidence", "free_decode_measured"))):
        raise ValueError("native factorization source identity or authority differs")
    measured = {scale: [verified_document(directory / "rows" / f"{scale:g}-{source}.json")
                        for source in plan["sources"]] for scale in plan["scales"]}
    rebuilt = independent_factor_admission(measured, plan["sources"])
    if any(report.get(key) != value for key, value in rebuilt.items()):
        raise ValueError("native factorization admission differs from independent source replay")
    drift = sorted(set(verified["current_implementation_drift"]) | {
        name for name, sha in report["implementation"].items()
        if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha})
    return {"report_receipt_sha256": report["receipt_sha256"],
            "candidate_step": verified["candidate_step"],
            "candidate_checkpoint_receipt_sha256": report["candidate_checkpoint_receipt_sha256"],
            "baseline_checkpoint_receipt_sha256": report["baseline_checkpoint_receipt_sha256"],
            "selected_scales": rebuilt["selected_scales"], "selected_totals": rebuilt["selected_totals"],
            "combination_count": len(rebuilt["adjudication"]),
            "current_implementation_drift": drift,
            "artifacts_verified": True, "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    result = verify_factorized_residual(args.report, args.training_directory)
    body = {"schema": "aura.native_factorized_residual_verification.v1", **result}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
