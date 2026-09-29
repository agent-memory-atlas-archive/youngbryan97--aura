#!/usr/bin/env python3
"""Adjudicate every source window of one matched 500-request development run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def complete_development(windows):
    """Require complete ordered coverage and one unchanged candidate and budget."""
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.verify_semantic_native_grammar import verified_source_window

    if not windows:
        raise ValueError("native development has no measured windows")
    excluded = {"plan_sha256", "sources", "source_window"}
    first = windows[0][0]
    common = {key: value for key, value in first.items() if key not in excluded}
    sources, outcomes, receipts = [], [], []
    for plan, comparison in windows:
        window = verified_source_window(plan, plan)
        if (window is None or plan.get("dataset") != "retained_validation"
                or plan.get("weight_mode") != "fitted"
                or plan.get("source_evidence") != "source_text"
                or {key: value for key, value in plan.items() if key not in excluded} != common
                or window["offset"] != len(sources)
                or window["ordered_sources_sha256"] != first["source_window"]["ordered_sources_sha256"]
                or comparison.get("training_plan_sha256") != plan["training_plan_sha256"]
                or comparison.get("checkpoint_receipt_sha256") != plan["checkpoint_receipt_sha256"]):
            raise ValueError("native development window coverage, candidate, or budget differs")
        intervention = comparison["source_intervention"]
        measured = comparison["comparison"]
        if (intervention["population"] != window["count"]
                or measured["population"] != window["count"]
                or [row["source_sha256"] for row in intervention["source_outcomes"]] != plan["sources"]
                or [row["source_sha256"] for row in measured["source_outcomes"]] != plan["sources"]):
            raise ValueError("native development comparison source coverage differs")
        sources.extend(plan["sources"])
        outcomes.extend(zip(measured["source_outcomes"], intervention["source_outcomes"], strict=True))
        receipts.append(comparison["receipt_sha256"])
    if (len(sources) != 500 or len(set(sources)) != 500
            or digest(sources) != first["source_window"]["ordered_sources_sha256"]):
        raise ValueError("native development requires all 500 distinct ordered requests")
    metrics = {}
    for name, field in (("procedure", "procedure"), ("answer", "answer")):
        metrics[name] = {"fitted_correct": sum(row["fitted_" + field] for row, _ in outcomes),
                         "base_correct": sum(row["base_" + field] for row, _ in outcomes),
                         "gains": sum(row["fitted_" + field] and not row["base_" + field]
                                      for row, _ in outcomes),
                         "regressions": sum(row["base_" + field] and not row["fitted_" + field]
                                            for row, _ in outcomes)}
    unforced = all(row["arms"]["fitted"]["decode_status"] == "completed"
                   and row["arms"]["fitted"]["bound_forced_completion"] is False
                   for _, row in outcomes)
    dependent = sum(row["gain_changes_under_erasure"] for _, row in outcomes)
    passed = (metrics["procedure"]["fitted_correct"] == 500
              and metrics["answer"]["fitted_correct"] == 500
              and all(metric["regressions"] == 0 for metric in metrics.values())
              and unforced and dependent > 0)
    return {"population": 500, "windows": len(windows), **metrics,
            "source_dependent_gains": dependent, "all_fitted_completions_unforced": unforced,
            "full_development_passed": passed,
            "current_stage": "fresh_transfer" if passed else "full_development",
            "training_plan_sha256": first["training_plan_sha256"],
            "checkpoint_receipt_sha256": first["checkpoint_receipt_sha256"],
            "ordered_sources_sha256": digest(sources), "comparison_receipts": receipts,
            "general_transfer_proven": False, "broad_gain_proven": False,
            "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", required=True, type=Path)
    parser.add_argument("--fit-verification", required=True, type=Path)
    parser.add_argument("--window-root", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from tools.adjudicate_semantic_native_micro_stages import verified_native_fit
    from tools.compare_semantic_native_grammar_fit import compare_directories
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    training, checkpoint, fit = verified_native_fit(args.training_directory, args.fit_verification)
    windows = []
    for root in args.window_root:
        plan = verified_document(root / "fitted/plan.json", "plan_sha256")
        comparison = compare_directories(fitted_directory=root / "fitted",
            base_directory=root / "base", erasure_directory=root / "erasure",
            training_directory=args.training_directory)
        windows.append((plan, comparison))
    result = complete_development(windows)
    if (result["training_plan_sha256"] != training["plan_sha256"]
            or result["checkpoint_receipt_sha256"] != checkpoint["receipt_sha256"]):
        raise ValueError("native development changed its verified fitted candidate")
    body = {"schema": "aura.native_development_progress.v1", **result,
            "fit_verification_receipt_sha256": fit["receipt_sha256"]}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    print(json.dumps(body, sort_keys=True))
    return 0 if result["full_development_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
