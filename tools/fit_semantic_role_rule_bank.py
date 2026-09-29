#!/usr/bin/env python3
"""Fit source-local role evidence from the native checkpoint's fit partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "aura.semantic_role_rule_fit.v1"


def _implementation() -> dict[str, str]:
    paths = ("core/learning/semantic_role_rule_induction.py",
             "tools/fit_semantic_role_rule_bank.py")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def fit_role_rule_bank(training_directory: Path, source_report_path: Path,
                       bundles: tuple[str, ...], *, minimum_constructions: int = 2) -> dict:
    """Refuse every source outside the checkpoint's declared fit partition."""
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_feature_materialization import (
        rebuild_semantic_feature_selection,
    )
    from core.learning.semantic_role_rule_induction import (
        induce_role_rules,
        observations_from_examples,
    )
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.refit_semantic_argument_proposals import source_bundle_arguments

    training = verified_document(training_directory / "plan.json", "plan_sha256")
    raw = source_report_path.read_bytes()
    source = json.loads(raw)
    if (training.get("schema") not in {"aura.semantic_native_fit_plan.v6",
                                        "aura.semantic_native_fit_plan.v7"}
            or training.get("held_labels_used_for_fit_or_selection") is not False
            or hashlib.sha256(raw).hexdigest() != training.get("source_report_sha256")
            or source.get("report_sha256") != _sha({key: value for key, value in source.items()
                                                    if key != "report_sha256"})):
        raise ValueError("role fit source report differs from native checkpoint")
    fit_ids = set(training["fit_ids"])
    other_ids = set(training["calibration_ids"]) | set(training["held_ids"])
    if not fit_ids or fit_ids & other_ids or not bundles:
        raise ValueError("role fit source partitions overlap or are empty")
    expected = source["representation_compatibility"]["source_feature_manifest_sha256s"]
    examples = []
    found = set()
    for argument in source_bundle_arguments(source, bundles=bundles):
        name, _, directory = argument.partition("=")
        manifest = json.loads((Path(directory) / "manifest.json").read_bytes())
        if manifest.get("manifest_sha256") != expected[name]:
            raise ValueError("role fit source manifest differs")
        _, reconstructed = rebuild_semantic_feature_selection(manifest)
        for example in reconstructed:
            identity = hashlib.sha256(example.source_text.encode()).hexdigest()
            if identity not in fit_ids:
                continue
            if example.split != "train" or identity in found:
                raise ValueError("role fit includes held or duplicate source")
            found.add(identity)
            examples.append(example)
    if found != fit_ids:
        raise ValueError("role fit partition is not fully reconstructed")
    observations = observations_from_examples(tuple(examples))
    bank = induce_role_rules(observations, minimum_constructions=minimum_constructions)
    if not bank.rules:
        raise ValueError("role fit found no cross-construction relation")
    body = {"schema": SCHEMA,
            "native_training_plan_sha256": training["plan_sha256"],
            "source_report_sha256": training["source_report_sha256"],
            "fit_source_sha256s": sorted(found),
            "fit_sources": len(examples), "observations": len(observations),
            "minimum_constructions": minimum_constructions,
            "implementation": _implementation(),
            "bank": bank.to_dict(), "serving_authority": False}
    return {**body, "receipt_sha256": digest(body)}


def read_role_rule_fit(path: Path, *, training: dict | None = None,
                       held_source_sha256s: tuple[str, ...] = ()) -> dict:
    """Verify the artifact and its disjoint training ancestry before scoring."""
    from core.learning.semantic_role_rule_induction import RoleRuleBank
    from tools.evaluate_semantic_native_checkpoint import digest

    envelope = json.loads(path.read_bytes())
    if (envelope.get("schema") != SCHEMA or envelope.get("schema_version") != 1
            or envelope.get("schema_name") != SCHEMA):
        raise ValueError("role fit envelope differs")
    value = envelope.get("payload")
    if (not isinstance(value, dict) or value.get("schema") != SCHEMA
            or value.get("receipt_sha256") != digest({key: item for key, item in value.items()
                                                      if key != "receipt_sha256"})
            or value.get("serving_authority") is not False
            or value.get("implementation") != _implementation()):
        raise ValueError("role fit receipt differs")
    RoleRuleBank.from_dict(value["bank"])
    fit_ids = value.get("fit_source_sha256s")
    if (not isinstance(fit_ids, list) or fit_ids != sorted(set(fit_ids))
            or len(fit_ids) != value.get("fit_sources")
            or set(fit_ids) & set(held_source_sha256s)):
        raise ValueError("role fit source inventory overlaps evaluation")
    if training is not None and (value["native_training_plan_sha256"] != training["plan_sha256"]
            or value["source_report_sha256"] != training["source_report_sha256"]
            or set(fit_ids) != set(training["fit_ids"])):
        raise ValueError("role fit ancestry differs from native checkpoint")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--minimum-constructions", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("role fit output already exists")
    payload = fit_role_rule_bank(args.training_directory, args.source_report,
        tuple(args.bundle), minimum_constructions=args.minimum_constructions)
    from core.runtime.file_write_gateway import get_file_write_gateway

    get_file_write_gateway().write_json(args.output, payload, schema_version=1,
        schema_name=SCHEMA, source="fit_semantic_role_rule_bank")
    read_role_rule_fit(args.output)
    print(json.dumps({"receipt_sha256": payload["receipt_sha256"],
                      "fit_sources": payload["fit_sources"],
                      "rules": len(payload["bank"]["rules"])}))


if __name__ == "__main__":
    main()
