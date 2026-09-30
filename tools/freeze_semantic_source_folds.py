#!/usr/bin/env python3
"""Freeze construction-disjoint source folds for a signed candidate cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def source_folds(examples, *, count: int = 3, seed: int = 0,
                 axis: str = "semantic"):
    from core.learning.semantic_construction_folds import (
        construction_folds,
        utterance_construction_folds,
    )

    train = tuple(item for item in examples if item.split == "train")
    if not train or len(train) == len(examples):
        raise ValueError("frozen source folds require train and withheld examples")
    if axis == "semantic":
        return construction_folds(train, count=count, seed=seed)
    if axis == "utterance":
        return utterance_construction_folds(train, count=count, seed=seed)
    raise ValueError("unsupported source fold axis")


def prepare_source_folds(model, report, bundles, *, candidate_report=None,
                         source_fit_only=False, count=3, seed=0, axis="semantic"):
    """Freeze training partitions without treating preparation as evaluation."""
    from core.learning.semantic_program_campaign import _sha
    from tools.refit_semantic_argument_proposals import (
        load_source_examples,
        verify_candidate_report_identity,
        verify_source_report_identity,
    )

    if type(source_fit_only) is not bool or source_fit_only == (candidate_report is not None):
        raise ValueError("choose either a measured candidate report or source-fit-only preparation")
    verify_source_report_identity(report, model)
    if source_fit_only and report.get("fit_complete") is not True:
        raise ValueError("source-fit-only preparation requires a completed source fit")
    if candidate_report is not None:
        verify_candidate_report_identity(candidate_report, model, report)
    folds = source_folds(load_source_examples(model, report, bundles),
                         count=count, seed=seed, axis=axis)
    body = {"schema": "aura.semantic_source_fold_preparation.v1",
            "candidate": model.receipt_sha256, "source_report_sha256": report["report_sha256"],
            "source_feature_manifest_sha256s": report["representation_compatibility"][
                "source_feature_manifest_sha256s"],
            "folds_receipt_sha256": folds["receipt_sha256"],
            "candidate_report_sha256": None if candidate_report is None
                else candidate_report["receipt_sha256"],
            "source_fit_only": source_fit_only, "validation_used": False, "test_used": False,
            "candidate_evaluation_performed": False, "qualification_evidence": False,
            "serving_authority": False}
    return folds, {**body, "receipt_sha256": _sha(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transducer", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    authority = parser.add_mutually_exclusive_group(required=True)
    authority.add_argument("--candidate-report", type=Path)
    authority.add_argument("--source-fit-only", action="store_true",
                           help="freeze source partitions before candidate evaluation; confers no qualification")
    parser.add_argument("--preparation-receipt", type=Path,
                        help="immutable model, source-cohort and fold custody; required for source-fit-only")
    parser.add_argument("--bundle", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--axis", choices=("semantic", "utterance"), default="semantic")
    args = parser.parse_args()
    if args.source_fit_only and args.preparation_receipt is None:
        parser.error("source-fit-only requires an explicit preparation receipt")
    if args.preparation_receipt is not None and args.preparation_receipt.resolve() == args.output.resolve():
        parser.error("folds and preparation receipt must be distinct")

    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from core.runtime.atomic_writer import atomic_write_bytes_if_absent
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        source_bundle_arguments,
    )

    configure_refit_environment(args.output)
    model = compositional_semantic_program_transducer_from_dict(
        json.loads(args.transducer.read_bytes()))
    report = json.loads(args.source_report.read_bytes())
    candidate_report = (None if args.candidate_report is None
                        else json.loads(args.candidate_report.read_bytes()))
    bundles = source_bundle_arguments(report, bundles=args.bundle)
    folds, preparation = prepare_source_folds(model, report, bundles,
        candidate_report=candidate_report, source_fit_only=args.source_fit_only,
        count=args.count, seed=args.seed, axis=args.axis)
    outputs = [(args.output, folds)]
    if args.preparation_receipt is not None:
        outputs.append((args.preparation_receipt, preparation))
    for path, document in outputs:
        payload = (json.dumps(document, sort_keys=True, separators=(",", ":"),
                              allow_nan=False) + "\n").encode("ascii")
        if (not atomic_write_bytes_if_absent(path, payload, mode=0o400)
                and path.read_bytes() != payload):
            raise ValueError("frozen source preparation differs from existing artifact")
    print(json.dumps({"output": str(args.output),
                      "receipt_sha256": folds["receipt_sha256"],
                      "population": len(folds["population"]),
                      "axis": args.axis,
                      "groups": folds.get("independent_groups", sum(
                          folds.get("family_construction_counts", {}).values())),
                      "source_fit_only": args.source_fit_only,
                      "preparation_receipt_sha256": preparation["receipt_sha256"],
                      "validation_used": False, "test_used": False,
                      "qualification_evidence": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
