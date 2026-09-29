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
    mode = fitted_plan.get("weight_mode")
    left, right = dict(fitted_plan), dict(base_plan)
    if mode in {"residual", "factorized"}:
        field = "residual_calibration" if mode == "residual" else "factorized_residual"
        schema = "v15" if left.get("schema", "").endswith(".v15") else (
            "v13" if left.get("schema", "").endswith(".v13") else
            "v9" if mode == "residual" else "v10")
        base_schema = "v8"
        if schema == "v15":
            base_schema = "v15"
        if schema == "v13":
            base_schema = ("v12" if left.get("source_window") is not None
                           else "v6" if left.get("dataset") in {"retained_validation", "retained_test"}
                           else "v11" if left.get("dataset") == "relation_transfer_controls"
                           else "v3")
        contract = left.pop(field, {})
        additions = {"tools/calibrate_semantic_native_residual.py",
                     "tools/verify_semantic_native_residual.py"}
        if mode == "factorized":
            additions |= {"core/learning/semantic_native_factorized_residual.py",
                          "tools/factor_semantic_native_residual.py",
                          "tools/verify_semantic_native_factorized_residual.py"}
        if (left.get("schema") != f"aura.semantic_native_grammar_plan.{schema}"
                or right.get("schema") != f"aura.semantic_native_grammar_plan.{base_schema}"
                or contract.get("baseline_checkpoint_receipt_sha256")
                    != right.get("checkpoint_receipt_sha256")
                or contract.get("source_only") is not True
                or contract.get("serving_authority") is not False
                or set(left.get("implementation", {})) - set(right.get("implementation", {}))
                    != additions):
            raise ValueError("native generation residual lineage differs")
        left["implementation"] = {key: value for key, value in left["implementation"].items()
                                  if key not in additions}
        excluded |= {"schema", "checkpoint_receipt_sha256"}
    if (mode not in {"fitted", "residual", "factorized"} or base_plan.get("weight_mode") != "base"
            or {key: value for key, value in left.items() if key not in excluded}
                != {key: value for key, value in right.items() if key not in excluded}):
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
    return {"population": len(sources), "candidate_weight_mode": mode, **metrics,
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


def matched_source_intervention(fitted_plan, base_plan, erasure_plan,
                                fitted_verification, base_verification, erasure_verification,
                                fitted_rows, base_rows, erasure_rows):
    """Grade one frozen decode cohort across weights and source-token erasure."""
    comparison = matched_generation(fitted_plan, base_plan,
                                    fitted_verification, base_verification)
    excluded = {"plan_sha256", "source_evidence"}
    if (fitted_plan.get("source_evidence") != "source_text"
            or base_plan.get("source_evidence") != "source_text"
            or erasure_plan.get("source_evidence") != "source_token_erasure"
            or {key: value for key, value in fitted_plan.items() if key not in excluded}
                != {key: value for key, value in erasure_plan.items() if key not in excluded}
            or erasure_verification.get("plan_sha256") != erasure_plan["plan_sha256"]
            or erasure_verification.get("weight_mode") != fitted_plan["weight_mode"]
            or erasure_verification.get("training_plan_sha256")
                != fitted_plan["training_plan_sha256"]
            or erasure_verification.get("checkpoint_receipt_sha256")
                != fitted_plan["checkpoint_receipt_sha256"]
            or erasure_verification.get("current_implementation_drift") != []
            or erasure_verification.get("artifacts_verified") is not True):
        raise ValueError("native source-intervention arms differ beyond erasure")
    sources = fitted_plan["sources"]
    if (fitted_plan.get("search_completions", 0) < 1
            or any(len(rows) != len(sources) for rows in (fitted_rows, base_rows, erasure_rows))):
        raise ValueError("native source-intervention search coverage differs")
    erasure_outcomes = erasure_verification.get("meaning_audit", {}).get("comparisons", [])
    if ([item.get("source_sha256") for item in erasure_outcomes] != sources
            or any(type(item.get("procedure_equivalent")) is not bool
                   for item in erasure_outcomes)):
        raise ValueError("native source-intervention erasure outcomes differ")
    outcomes = []
    for source, pair, erased, fitted_row, base_row, erasure_row in zip(
            sources, comparison["source_outcomes"], erasure_outcomes,
            fitted_rows, base_rows, erasure_rows, strict=True):
        if (any(row.get("source_sha256") != source
                or not isinstance(row.get("search"), dict)
                or type(row["search"].get("requested_top_k_proven")) is not bool
                or type(row["search"].get("observed_program_reach")) is not bool
                for row in (fitted_row, base_row, erasure_row))
                or any(row.get("decode_status") not in
                       {"completed", "search_without_completion"}
                       for row in (fitted_row, base_row, erasure_row))):
            raise ValueError("native source-intervention row inventory differs")
        gain = (pair["fitted_procedure"] and pair["fitted_answer"]
                and not pair["base_procedure"]
                and not fitted_row["bound_forced_completion"])
        selection_changed = fitted_row["program"] != erasure_row["program"]
        lost_under_erasure = not erased["procedure_equivalent"]
        outcomes.append({"source_sha256": source,
            "fitted_exact": pair["fitted_procedure"],
            "base_exact": pair["base_procedure"],
            "erasure_exact": erased["procedure_equivalent"],
            "exact_gain": gain,
            "exact_regression": pair["base_procedure"] and not pair["fitted_procedure"],
            "gain_changes_under_erasure": gain and (selection_changed or lost_under_erasure),
            "gain_loses_exactness_under_erasure": gain and lost_under_erasure,
            "arms": {name: {"decode_status": row["decode_status"],
                "candidate_count": len(row["search"]["proposals"]),
                "observed_program_reach": row["search"]["observed_program_reach"],
                "requested_top_k_proven": row["search"]["requested_top_k_proven"],
                "halt_reason": row["search"]["halt_reason"],
                "bound_forced_completion": row["bound_forced_completion"]}
                for name, row in (("fitted", fitted_row), ("base", base_row),
                                  ("erasure", erasure_row))}})
    gains = sum(row["exact_gain"] for row in outcomes)
    regressions = sum(row["exact_regression"] for row in outcomes)
    source_dependent_gains = sum(row["gain_changes_under_erasure"] for row in outcomes)
    return {"population": len(sources), "exact_gains": gains,
            "exact_regressions": regressions,
            "source_dependent_gains": source_dependent_gains,
            "baseline_regression_control_informative":
                any(row["base_exact"] for row in outcomes),
            "bounded_micro_probe_passed": gains > 0 and regressions == 0
                and source_dependent_gains > 0,
            "source_outcomes": outcomes, "general_transfer_proven": False,
            "broad_gain_proven": False, "serving_authority": False}


def compare_directories(*, fitted_directory, base_directory, training_directory,
                        erasure_directory=None, relation_controls_directory=None):
    """Recompute the comparison from independently verified durable arms."""
    from types import SimpleNamespace

    args = SimpleNamespace(fitted_directory=fitted_directory, base_directory=base_directory,
        training_directory=training_directory, erasure_directory=erasure_directory,
        relation_controls_directory=relation_controls_directory)
    if relation_controls_directory is not None and erasure_directory is None:
        raise ValueError("relation controls require the preceding matched source-erasure stage")
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.verify_semantic_native_grammar import (
        audit_grammar_meanings,
        verified_dataset,
        verified_examples,
        verify_grammar,
    )

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
    erasure = None
    erasure_verification = None
    if args.erasure_directory is not None:
        verified = verify_grammar(args.erasure_directory, args.training_directory)
        plan = verified_document(args.erasure_directory / "plan.json", "plan_sha256")
        report = verified_document(args.erasure_directory / "report.json")
        dataset, seed = verified_dataset(plan, report)
        examples = verified_examples(plan, dataset=dataset, seed=seed)
        rows = [verified_document(args.erasure_directory / "rows" / f"{source}.json")
                for source in plan["sources"]]
        verified["meaning_audit"] = audit_grammar_meanings(rows, examples)
        arm_rows = [[verified_document(directory / "rows" / f"{source}.json")
                     for source in plan["sources"]]
                    for directory, plan in zip(
                        (args.fitted_directory, args.base_directory), plans, strict=True)]
        erasure = matched_source_intervention(
            *plans, plan, *verifications, verified, *arm_rows, rows)
        erasure_verification = verified
    schema = ("aura.native_grammar_fit_comparison.v3" if erasure is not None
              else "aura.native_grammar_fit_comparison.v1" if plans[0]["weight_mode"] == "fitted"
              else "aura.native_grammar_fit_comparison.v2")
    body = {"schema": schema,
            "fitted_report_receipt_sha256": verifications[0]["report_receipt_sha256"],
            "base_report_receipt_sha256": verifications[1]["report_receipt_sha256"],
            "training_plan_sha256": plans[0]["training_plan_sha256"],
            "checkpoint_receipt_sha256": plans[0]["checkpoint_receipt_sha256"],
            "comparison": matched_generation(*plans, *verifications)}
    if erasure is not None:
        body["erasure_report_receipt_sha256"] = erasure_verification["report_receipt_sha256"]
        body["source_intervention"] = erasure
    if args.relation_controls_directory is not None:
        from tools.semantic_native_relation_transfer import adjudicate_native_relation_transfer

        controlled_verification = verify_grammar(args.relation_controls_directory,
                                                args.training_directory)
        controlled_plan = verified_document(args.relation_controls_directory / "plan.json", "plan_sha256")
        controlled_rows = [verified_document(args.relation_controls_directory / "rows" / f"{source}.json")
                           for source in controlled_plan["sources"]]
        reference_rows = [verified_document(args.fitted_directory / "rows" / f"{source}.json")
                          for source in plans[0]["sources"]]
        body["schema"] = "aura.native_grammar_fit_comparison.v4"
        body["relation_controls_report_receipt_sha256"] = controlled_verification["report_receipt_sha256"]
        body["relation_transfer"] = adjudicate_native_relation_transfer(
            plans[0], controlled_plan, verifications[0], controlled_verification,
            reference_rows, controlled_rows)
    return {**body, "receipt_sha256": digest(body)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted-directory", required=True, type=Path)
    parser.add_argument("--base-directory", required=True, type=Path)
    parser.add_argument("--training-directory", required=True, type=Path)
    parser.add_argument("--erasure-directory", type=Path)
    parser.add_argument("--relation-controls-directory", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.relation_controls_directory is not None and args.erasure_directory is None:
        parser.error("relation controls require the preceding matched source-erasure stage")
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    result = compare_directories(fitted_directory=args.fitted_directory,
        base_directory=args.base_directory, training_directory=args.training_directory,
        erasure_directory=args.erasure_directory,
        relation_controls_directory=args.relation_controls_directory)
    _save_if_absent(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
