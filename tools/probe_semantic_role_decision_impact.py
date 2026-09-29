#!/usr/bin/env python3
"""Replay a role prior against measured calibration choices without model load.

This is a conditional teacher-path diagnostic, not free decoding. Target
operations and correct indices are used only after scoring to count effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "aura.semantic_role_decision_impact.v1"


def adjusted_teacher_path(bank, source: str, decisions: list[dict],
                          *, input_count: int, strength: float) -> tuple[list[dict], int]:
    """Change only the first operation's two reference competitions."""
    from core.learning.semantic_native_grammar import NativeGrammarDecision
    from core.learning.semantic_role_rule_induction import native_role_rule_adjustments

    if (len(decisions) < 3 or decisions[0]["kind"] != "operation"
            or [row["kind"] for row in decisions[1:3]] != ["reference", "reference"]):
        raise ValueError("calibration row lacks a binary first operation")
    operation = decisions[0]["choices"][decisions[0]["correct_index"]]
    adjusted = list(decisions)
    affected = 0
    for role in range(2):
        row = decisions[role + 1]
        choices = tuple(NativeGrammarDecision("", (0, 0), value, 0, role, operation)
                        for value in row["choices"])
        delta = native_role_rule_adjustments(bank, source, choices,
            input_count=input_count, strength=strength)
        if any(delta):
            affected += 1
            adjusted[role + 1] = {**row, "scores": [score + change for score, change
                                                   in zip(row["scores"], delta, strict=True)]}
    return adjusted, affected


def calibration_impact(bank, examples: dict[str, object], rows: dict[str, dict],
                       *, strength: float) -> dict:
    """Count exact-path gains and losses on the same saved source scores."""
    from core.learning.semantic_native_path_calibration import native_path_profile
    from core.learning.semantic_public_inputs import semantic_public_character_inputs

    if set(examples) != set(rows):
        raise ValueError("calibration examples and score rows differ")
    details = []
    for identity in sorted(rows):
        row = rows[identity]
        example = examples[identity]
        source = example.source_text
        original = row["decisions"]
        revised, affected = adjusted_teacher_path(bank, source, original,
            input_count=len(semantic_public_character_inputs(source).literals),
            strength=strength)
        before = native_path_profile(original)
        after = native_path_profile(revised)
        details.append({"source_sha256": identity, "adjusted_reference_competitions": affected,
                        "baseline_exact_teacher_path": before["exact_teacher_path"],
                        "candidate_exact_teacher_path": after["exact_teacher_path"],
                        "baseline_first_error": before["first_error"],
                        "candidate_first_error": after["first_error"]})
    return {"population": len(details),
            "affected_sources": sum(row["adjusted_reference_competitions"] > 0
                                    for row in details),
            "affected_reference_competitions": sum(row["adjusted_reference_competitions"]
                                                   for row in details),
            "new_exact_teacher_paths": sum(not row["baseline_exact_teacher_path"]
                and row["candidate_exact_teacher_path"] for row in details),
            "lost_exact_teacher_paths": sum(row["baseline_exact_teacher_path"]
                and not row["candidate_exact_teacher_path"] for row in details),
            "rows": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--calibration-directory", type=Path, required=True)
    parser.add_argument("--role-rule-bank", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--strength", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("role decision impact output already exists")

    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_feature_materialization import (
        rebuild_semantic_feature_selection,
    )
    from core.learning.semantic_role_rule_induction import RoleRuleBank
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.fit_semantic_role_rule_bank import read_role_rule_fit
    from tools.refit_semantic_argument_proposals import source_bundle_arguments
    from tools.verify_semantic_native_residual import verify_residual

    training = verified_document(args.training_directory / "plan.json", "plan_sha256")
    residual = verify_residual(args.calibration_directory, args.training_directory)
    scale = residual["selected_scale"]
    plan = verified_document(args.calibration_directory / "plan.json", "plan_sha256")
    if plan["sources"] != sorted(training["calibration_ids"]):
        raise ValueError("calibration population differs from training")
    fit = read_role_rule_fit(args.role_rule_bank, training=training,
                             held_source_sha256s=tuple(plan["sources"]))
    bank = RoleRuleBank.from_dict(fit["bank"])
    source_bytes = args.source_report.read_bytes()
    source_report = json.loads(source_bytes)
    if (hashlib.sha256(source_bytes).hexdigest() != training["source_report_sha256"]
            or source_report["report_sha256"] != _sha({key: value for key, value
                                                       in source_report.items()
                                                       if key != "report_sha256"})):
        raise ValueError("calibration source report differs")
    expected_manifests = source_report["representation_compatibility"][
        "source_feature_manifest_sha256s"]
    examples = {}
    for argument in source_bundle_arguments(source_report, bundles=tuple(args.bundle)):
        name, _, directory = argument.partition("=")
        manifest = json.loads((Path(directory) / "manifest.json").read_bytes())
        if manifest.get("manifest_sha256") != expected_manifests[name]:
            raise ValueError("calibration source manifest differs")
        _, reconstructed = rebuild_semantic_feature_selection(manifest)
        for example in reconstructed:
            identity = hashlib.sha256(example.source_text.encode()).hexdigest()
            if identity not in training["calibration_ids"]:
                continue
            if identity in examples:
                raise ValueError("duplicate calibration source")
            examples[identity] = example
    rows = {identity: verified_document(args.calibration_directory / "rows" /
            f"{scale:g}-{identity}.json") for identity in plan["sources"]}
    impact = calibration_impact(bank, examples, rows, strength=args.strength)
    body = {"schema": SCHEMA, "role_fit_receipt_sha256": fit["receipt_sha256"],
            "calibration_report_receipt_sha256": residual["report_receipt_sha256"],
            "historical_training_implementation_drift": residual["current_implementation_drift"],
            "calibration_scale": scale, "strength": args.strength,
            "training_plan_sha256": training["plan_sha256"], **impact,
            "target_available_to_scorer": False, "free_decode_measured": False,
            "general_transfer_proven": False, "serving_authority": False}
    payload = {**body, "receipt_sha256": digest(body)}
    get_file_write_gateway().write_json(args.output, payload, schema_version=1,
        schema_name=SCHEMA, source="probe_semantic_role_decision_impact")
    print(json.dumps({key: payload[key] for key in ("receipt_sha256", "population",
        "affected_sources", "affected_reference_competitions",
        "new_exact_teacher_paths", "lost_exact_teacher_paths")}))


if __name__ == "__main__":
    main()
