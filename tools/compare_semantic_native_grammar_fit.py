#!/usr/bin/env python3
"""Compare verified fitted/base generation without confusing procedure and meaning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def matched_generation(fitted_plan, base_plan, fitted_verification, base_verification):
    excluded = {"weight_mode", "plan_sha256"}
    if (fitted_plan.get("weight_mode") != "fitted" or base_plan.get("weight_mode") != "base"
            or {key: value for key, value in fitted_plan.items() if key not in excluded}
                != {key: value for key, value in base_plan.items() if key not in excluded}):
        raise ValueError("native generation arms differ beyond fitted weights")
    sources = fitted_plan.get("sources")
    if not isinstance(sources, list) or not sources or len(set(sources)) != len(sources):
        raise ValueError("native generation source population differs")
    arms = []
    for plan, verification in ((fitted_plan, fitted_verification), (base_plan, base_verification)):
        if (verification.get("plan_sha256") != plan["plan_sha256"]
                or verification.get("current_implementation_drift") != []
                or verification.get("artifacts_verified") is not True
                or verification.get("weight_mode") != plan["weight_mode"]
                or verification.get("training_plan_sha256") != plan.get("training_plan_sha256")
                or verification.get("checkpoint_receipt_sha256") != plan.get("checkpoint_receipt_sha256")):
            raise ValueError("native generation verification identity differs")
        rows = verification.get("meaning_audit", {}).get("comparisons", [])
        if ([row.get("source_sha256") for row in rows] != sources
                or any(type(row.get(key)) is not bool for row in rows
                       for key in ("procedure_equivalent", "observed_answer_correct"))
                or any(row.get("meaning", {}).get("status") not in
                       {"equivalent", "different", "unknown", "unavailable"} for row in rows)):
            raise ValueError("native generation outcome coverage differs")
        arms.append(rows)
    pairs = list(zip(*arms, strict=True))
    metrics = {}
    for name, key in (("procedure", "procedure_equivalent"), ("answer", "observed_answer_correct")):
        metrics[name] = {"fitted_correct": sum(left[key] for left, _ in pairs),
                         "base_correct": sum(right[key] for _, right in pairs),
                         "gains": sum(left[key] and not right[key] for left, right in pairs),
                         "regressions": sum(right[key] and not left[key] for left, right in pairs)}
    return {"population": len(sources), **metrics,
            "source_outcomes": [{"source_sha256": source,
                "fitted_procedure": left["procedure_equivalent"], "base_procedure": right["procedure_equivalent"],
                "fitted_answer": left["observed_answer_correct"], "base_answer": right["observed_answer_correct"],
                "fitted_meaning": left["meaning"]["status"], "base_meaning": right["meaning"]["status"]}
                for source, (left, right) in zip(sources, pairs, strict=True)],
            "new_proven_meanings": sum(left["meaning"]["status"] == "equivalent"
                and right["meaning"]["status"] != "equivalent" for left, right in pairs),
            "lost_proven_meanings": sum(right["meaning"]["status"] == "equivalent"
                and left["meaning"]["status"] != "equivalent" for left, right in pairs),
            "base_proven_fitted_witnessed_different": sum(right["meaning"]["status"] == "equivalent"
                and left["meaning"]["status"] == "different" for left, right in pairs),
            "general_transfer_proven": False, "broad_gain_proven": False, "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted-directory", required=True, type=Path)
    parser.add_argument("--base-directory", required=True, type=Path)
    parser.add_argument("--training-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.verify_semantic_native_grammar import (
        audit_grammar_meanings,
        verified_dataset,
        verified_examples,
        verify_grammar,
    )

    configure_refit_environment(args.output)
    plans, verifications = [], []
    for directory in (args.fitted_directory, args.base_directory):
        verified = verify_grammar(directory, args.training_directory)
        plan = verified_document(directory / "plan.json", "plan_sha256")
        report = verified_document(directory / "report.json")
        dataset, seed = verified_dataset(plan, report)
        examples = verified_examples(plan, dataset=dataset, seed=seed)
        rows = [verified_document(directory / "rows" / f"{source}.json") for source in plan["sources"]]
        verified["meaning_audit"] = audit_grammar_meanings(rows, examples)
        plans.append(plan)
        verifications.append(verified)
    body = {"schema": "aura.native_grammar_fit_comparison.v1",
            "fitted_report_receipt_sha256": verifications[0]["report_receipt_sha256"],
            "base_report_receipt_sha256": verifications[1]["report_receipt_sha256"],
            "training_plan_sha256": plans[0]["training_plan_sha256"],
            "checkpoint_receipt_sha256": plans[0]["checkpoint_receipt_sha256"],
            "comparison": matched_generation(*plans, *verifications)}
    result = {**body, "receipt_sha256": digest(body)}
    _save_if_absent(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
