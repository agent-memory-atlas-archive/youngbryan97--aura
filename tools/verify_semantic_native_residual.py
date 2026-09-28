#!/usr/bin/env python3
"""Independently regrade a source-only native LoRA residual calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learning.semantic_native_path_calibration import (  # noqa: E402
    native_path_profile,
    native_path_totals,
)
from tools.audit_semantic_native_path_calibration import calibration_competitions  # noqa: E402
from tools.evaluate_semantic_native_checkpoint import (  # noqa: E402
    digest,
    selected_checkpoint,
    verified_document,
)


def verify_residual(directory: Path, training_directory: Path) -> dict:
    training, selected = selected_checkpoint(training_directory)
    training_report = verified_document(training_directory / "report.json")
    supervision = verified_document(training_directory / "supervision.json")
    competitions = calibration_competitions(training, supervision)
    plan = verified_document(directory / "plan.json", "plan_sha256")
    report = verified_document(directory / "report.json")
    scales, sources = plan["scales"], sorted(competitions)
    if (plan.get("schema") != "aura.native_residual_calibration_plan.v1"
            or plan.get("training_plan_sha256") != training["plan_sha256"]
            or plan.get("training_report_receipt_sha256") != training_report["receipt_sha256"]
            or plan.get("baseline_checkpoint_receipt_sha256") != selected["receipt_sha256"]
            or plan.get("supervision_receipt_sha256") != supervision["receipt_sha256"]
            or selected["step"] != 0 or plan.get("sources") != sources
            or not isinstance(scales, list) or len(scales) < 3
            or sorted(set(scales)) != scales or scales[0] != 0. or scales[-1] != 1.
            or any(type(scale) not in {int, float} or not math.isfinite(scale)
                   or not 0. <= scale <= 1. for scale in scales)
            or plan.get("held_labels_used") is not False
            or plan.get("training_selection_unchanged") is not True
            or plan.get("serving_authority") is not False
            or report.get("schema") != "aura.native_residual_calibration.v1"
            or report.get("plan_sha256") != plan["plan_sha256"]
            or report.get("held_labels_used") is not False
            or report.get("training_selection_unchanged") is not True
            or report.get("serving_authority") is not False
            or report.get("general_transfer_proven") is not False):
        raise ValueError("residual calibration source protocol differs")
    candidate = [row for row in training_report["checkpoints"]
                 if row["receipt_sha256"] == plan["candidate_checkpoint_receipt_sha256"]]
    if len(candidate) != 1 or candidate[0]["step"] == 0:
        raise ValueError("residual candidate is not a measured fitted checkpoint")
    endpoint = {0.: verified_document(training_directory / "calibration-paths-0.json"),
                1.: verified_document(training_directory /
                                     f"calibration-paths-{candidate[0]['step']}.json")}
    if (endpoint[0.]["receipt_sha256"] != selected["calibration_path_receipt_sha256"]
            or endpoint[1.]["receipt_sha256"] != candidate[0]["calibration_path_receipt_sha256"]):
        raise ValueError("residual endpoint receipts differ from checkpoints")
    endpoint_rows = {scale: {row["source"]: row for row in document["rows"]}
                     for scale, document in endpoint.items()}
    expected_names = {f"{scale:g}-{source}.json" for scale in scales for source in sources}
    if {path.name for path in (directory / "rows").glob("*.json")} != expected_names:
        raise ValueError("residual calibration row inventory differs")
    rows_by_scale, receipts, totals, exact = {}, {}, {}, {}
    for scale in scales:
        rows, receipts[str(scale)] = [], {}
        for source in sources:
            row = verified_document(directory / "rows" / f"{scale:g}-{source}.json")
            if (row.get("schema") != "aura.native_residual_calibration_row.v1"
                    or row.get("plan_sha256") != plan["plan_sha256"]
                    or row.get("source") != source or row.get("scale") != scale
                    or len(row.get("decisions", [])) != len(competitions[source])):
                raise ValueError("residual calibration row identity differs")
            for measured, alternatives in zip(row["decisions"], competitions[source], strict=True):
                if (measured.get("kind") != alternatives[0]["kind"]
                        or measured.get("correct_index") != alternatives[0]["correct_index"]
                        or measured.get("choices") != [item["choice"] for item in alternatives]
                        or len(measured.get("scores", [])) != len(alternatives)):
                    raise ValueError("residual calibration alternatives differ")
            profile = native_path_profile(row["decisions"])
            if row.get("profile") != profile:
                raise ValueError("residual calibration path profile differs")
            if scale in endpoint_rows:
                previous = endpoint_rows[scale][source]
                errors = [abs(actual - expected)
                          for decision, old in zip(row["decisions"], previous["decisions"], strict=True)
                          for actual, expected in zip(decision["scores"], old["scores"], strict=True)]
                if (max(errors, default=0.) > 0.015625
                        or profile != native_path_profile(previous["decisions"])):
                    raise ValueError("residual endpoint scores differ from measured checkpoint")
            rows.append(row)
            receipts[str(scale)][source] = row["receipt_sha256"]
        rows_by_scale[scale] = rows
        totals[str(scale)] = native_path_totals(rows, sources)
        exact[scale] = {row["source"] for row in rows if row["profile"]["exact_teacher_path"]}
    baseline = exact[0.]
    adjudication = [{"scale": scale, "exact_teacher_paths": totals[str(scale)]["exact_teacher_paths"],
                     "lost_baseline_sources": sorted(baseline - exact[scale]),
                     "new_exact_sources": sorted(exact[scale] - baseline),
                     "eligible": baseline <= exact[scale]} for scale in scales]
    selected_scale = min((row for row in adjudication if row["eligible"]),
                         key=lambda row: (-row["exact_teacher_paths"], row["scale"]))["scale"]
    if (report.get("totals") != totals or report.get("adjudication") != adjudication
            or report.get("selected_scale") != selected_scale
            or report.get("row_receipts") != receipts):
        raise ValueError("residual calibration selection or totals differ")
    return {"artifacts_verified": True,
            "training_plan_sha256": training["plan_sha256"],
            "training_report_receipt_sha256": training_report["receipt_sha256"],
            "report_receipt_sha256": report["receipt_sha256"],
            "selected_scale": selected_scale,
            "candidate_step": candidate[0]["step"],
            "totals": totals,
            "baseline_paths_lost_at_selected_scale": len(baseline - exact[selected_scale]),
            "new_paths_at_selected_scale": len(exact[selected_scale] - baseline),
            "source_only": True, "model_scores_independently_recomputed": False,
            "free_decode_measured": False, "general_transfer_proven": False,
            "serving_authority": False,
            "current_implementation_drift": sorted(name for name, sha in
                plan["implementation"].items() if not (ROOT / name).is_file()
                or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    result = verify_residual(args.directory, args.training_directory)
    body = {"schema": "aura.native_residual_calibration_verification.v1", **result}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
