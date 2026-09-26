#!/usr/bin/env python3
"""Regrade a complete native fit from source features in a separate CPU process."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def verify_native_totals(report, rows):
    """Recompute reported counts without treating unknown labels as successes."""
    identities = [row["source"] for row in rows]
    if len(set(identities)) != len(identities) or report["rows"] != rows:
        raise ValueError("native fit durable rows differ or repeat")
    totals = {"population": len(rows),
        "incumbent_correct": sum(row["incumbent_correct"] is True for row in rows),
        "native_correct": sum(row["selected_correct"] is True for row in rows),
        "pretrained_correct": sum(row["pretrained_correct"] is True for row in rows),
        "bank_reachable": sum(row["bank_reachable"] is True for row in rows),
        "gains": sum(row["selected_correct"] is True and row["incumbent_correct"] is False for row in rows),
        "regressions": sum(row["selected_correct"] is False and row["incumbent_correct"] is True for row in rows),
        "learning_gains": sum(row["selected_correct"] is True and row["pretrained_correct"] is False for row in rows),
        "learning_regressions": sum(row["selected_correct"] is False and row["pretrained_correct"] is True for row in rows)}
    if any(type(report[key]) is not int or report[key] != value for key, value in totals.items()):
        raise ValueError("native fit reported totals differ from durable rows")
    return {**totals, "unknown_selected": sum(row["selected_correct"] is None for row in rows),
            "unknown_pretrained": sum(row["pretrained_correct"] is None for row in rows)}


def regrade_bank(bank, item):
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_graph_counterexamples import ProgramObservationCache, compare_program_meanings, counterfactual_inputs
    from core.learning.semantic_joint_graph_learning import align_source_input_registers
    from core.learning.semantic_program_ir import TokenSpan

    spans = tuple(TokenSpan(**span) for span in bank["bank"]["input_spans"])
    aligned, _ = align_source_input_registers(item, spans)
    target = Program(len(item.public_inputs), tuple(Instruction(step.op, step.args) for step in aligned))
    if target.sha() != bank["diagnosis"]["target_program_sha256"]:
        raise ValueError("native bank target differs from source feature grounding")
    labels, counts, cache = {}, Counter(), ProgramObservationCache()
    recorded = {row["program_sha256"]: row["status"] for row in bank["diagnosis"]["comparisons"]}
    if len(recorded) != len(bank["diagnosis"]["comparisons"]):
        raise ValueError("native bank semantic comparisons repeat")
    for candidate in bank["bank"]["candidates"]:
        key = candidate["program_sha256"]
        if key in labels:
            continue
        payload = candidate["program"]
        program = Program(len(spans), tuple(Instruction(op, tuple(args)) for op, args in payload["instructions"]))
        if program.sha() != key or payload["sha"] != key:
            raise ValueError("native bank program digest differs")
        comparison = compare_program_meanings(target, program, counterfactual_inputs(item.public_inputs), observation_cache=cache)
        if recorded.get(key) != comparison["status"]:
            raise ValueError("native bank semantic status differs from independent execution")
        counts[comparison["status"]] += 1
        labels[key] = {"equivalent": True, "different": False, "unknown": None}[comparison["status"]]
    if set(labels) != set(recorded):
        raise ValueError("native bank comparison inventory differs from executable proposals")
    return labels, dict(counts)


def verify_fit(directory, bank_directory, items):
    from tools.evaluate_semantic_candidate_ranker import _read_bank
    from tools.evaluate_semantic_native_checkpoint import selected_checkpoint, verified_document, verify_replay_row
    from tools.train_nested_semantic_ranker import _verified_pair

    plan, selected = selected_checkpoint(directory)
    report = verified_document(directory / "report.json")
    bank_plan, bank_report = _verified_pair(bank_directory)
    if (report.get("schema") != "aura.semantic_native_fit.v1"
            or any(report.get(key) is not False for key in ("serving_authority", "qualification_evidence", "held_labels_used_for_fit_or_selection"))
            or plan["bank_plan_sha256"] != bank_plan["plan_sha256"]
            or plan["bank_receipt_sha256"] != bank_report["receipt_sha256"]
            or not set(plan["held_ids"]) <= set(bank_plan["held_ids"])):
        raise ValueError("native fit and source bank authority differ")
    history = report["history"]
    if (len(history) != plan["steps"]
            or any(row["step"] != index or not math.isfinite(row["loss"]) for index, row in enumerate(history, 1))):
        raise ValueError("native fit optimizer history is incomplete")
    checkpoints = [verified_document(directory / f"checkpoint-{row['step']}.json") for row in report["checkpoints"]]
    if checkpoints != report["checkpoints"] or selected not in checkpoints:
        raise ValueError("native fit checkpoint report differs")
    supervision = verified_document(directory / "supervision.json")
    if (supervision["plan_sha256"] != plan["plan_sha256"]
            or supervision["receipt_sha256"] != report["supervision_receipt_sha256"]
            or len(supervision["rows"]) != report["prefix_sequence_population"]
            or {row["source"] for row in supervision["rows"]} != set(plan["captured_fit_ids"]) | set(plan["calibration_ids"])
            or report["gradient_source_population"] != len(set(plan["scheduled_fit_ids"]))):
        raise ValueError("native fit supervision coverage differs")
    rows, statuses = [], Counter()
    for identity in plan["held_ids"]:
        row = verified_document(directory / "rows" / f"{identity}.json")
        bank = _read_bank(bank_directory / "rows" / f"{identity}.json", source=identity,
            plan_sha=bank_plan["plan_sha256"], model_receipt=bank_report["candidate_receipt_sha256"],
            expected_receipt=bank_report["row_receipts"][identity])
        labels, counts = regrade_bank(bank, items[identity])
        statuses.update(counts)
        keys = row["program_sha256s"]
        incumbent = bank["bank"]["selected_program_sha256"]
        if incumbent is not None and (not keys or keys[0] != incumbent):
            raise ValueError("native fit changed the incumbent proposal identity")
        verify_replay_row(row, source=identity, plan_sha256=plan["plan_sha256"], programs=keys,
            labels=tuple(labels[key] for key in keys), incumbent_available=incumbent is not None)
        if set(keys) != set(labels):
            raise ValueError("native fit omitted or added source-bank proposals")
        rows.append(row)
    totals = verify_native_totals(report, rows)
    differences = sorted(name for name, sha in plan["implementation"].items()
                         if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha)
    return {"training_plan_sha256": plan["plan_sha256"], "training_receipt_sha256": report["receipt_sha256"],
            "selected_checkpoint_receipt_sha256": selected["receipt_sha256"], "selected_step": selected["step"],
            "totals": totals, "semantic_status_counts": dict(statuses),
            "current_implementation_drift": differences, "artifacts_verified": True,
            "general_transfer_proven": False, "broad_gain_proven": False, "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "bank", "parent", "source-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment, load_source_examples, source_bundle_arguments
    from tools.train_nested_semantic_ranker import _verified_pair
    from core.learning.semantic_program_compositional_transducer import compositional_semantic_program_transducer_from_dict

    configure_refit_environment(args.output)
    bank_plan, _ = _verified_pair(args.bank)
    parent = compositional_semantic_program_transducer_from_dict(json.loads(args.parent.read_bytes()))
    raw_source = args.source_report.read_bytes()
    if (parent.receipt_sha256 != bank_plan["parent_receipt_sha256"]
            or hashlib.sha256(raw_source).hexdigest() != bank_plan["source_report_sha256"]):
        raise ValueError("native verification source basis differs")
    source = json.loads(raw_source)
    examples = load_source_examples(parent, source, source_bundle_arguments(source, bundles=args.bundle))
    items = {item.ir.source_text_sha256: item for item in examples}
    result = verify_fit(args.directory, args.bank, items)
    body = {"schema": "aura.semantic_native_fit_verification.v1", **result}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    verified_document(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
