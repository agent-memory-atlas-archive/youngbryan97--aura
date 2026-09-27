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


def compare_source_erasure_outcomes(reference_plan, control_plan, reference_report, control_report):
    """Pair a matched fit control without equating aggregate ties to equal programs."""
    from core.learning.semantic_native_source_control import source_control_mode_from_plan
    from core.learning.semantic_program_replication import _exact_one_sided_pair

    if (source_control_mode_from_plan(reference_plan) != "source_text"
            or source_control_mode_from_plan(control_plan) != "source_token_erasure"):
        raise ValueError("native fit comparison needs intact and erased fitting arms")
    allowed = {"schema", "plan_sha256", "implementation", "input", "source_evidence_control"}
    left = {key: value for key, value in reference_plan.items() if key not in allowed}
    right = {key: value for key, value in control_plan.items() if key not in allowed}
    if left != right:
        raise ValueError("native source control changes the matched fit protocol")
    for plan, report in ((reference_plan, reference_report), (control_plan, control_report)):
        if report["plan_sha256"] != plan["plan_sha256"]:
            raise ValueError("native fit comparison report belongs to another plan")
        verify_native_totals(report, report["rows"])
    reference = {row["source"]: row for row in reference_report["rows"]}
    control = {row["source"]: row for row in control_report["rows"]}
    if set(reference) != set(reference_plan["held_ids"]) or reference.keys() != control.keys():
        raise ValueError("native fit comparison source population differs")
    baseline_fields = ("program_sha256s", "incumbent_correct", "pretrained_correct",
                       "pretrained_program_sha256", "bank_reachable")
    if any(any(reference[key].get(field) != control[key].get(field) for field in baseline_fields)
           for key in reference):
        raise ValueError("native fit comparison proposal bank or baseline differs")
    known = [key for key in sorted(reference)
             if type(reference[key]["selected_correct"]) is bool
             and type(control[key]["selected_correct"]) is bool]
    paired = _exact_one_sided_pair(
        [{"source_text_sha256": key, "answer_exact": reference[key]["selected_correct"]} for key in known],
        [{"source_text_sha256": key, "answer_exact": control[key]["selected_correct"]} for key in known],
        metric="answer_exact") if known else None
    return {"reference_plan_sha256": reference_plan["plan_sha256"],
            "control_plan_sha256": control_plan["plan_sha256"],
            "reference_report_sha256": reference_report["receipt_sha256"],
            "control_report_sha256": control_report["receipt_sha256"],
            "population": len(reference), "known_pairs": len(known),
            "unknown_pairs": len(reference) - len(known),
            "both_correct": sum(reference[key]["selected_correct"] and control[key]["selected_correct"]
                                for key in known),
            "both_incorrect": sum(not reference[key]["selected_correct"]
                                  and not control[key]["selected_correct"] for key in known),
            "different_selected_programs": sum(reference[key]["chosen_program_sha256"]
                                               != control[key]["chosen_program_sha256"] for key in reference),
            "paired_exact_test": paired,
            "implementation_differences": sorted(name for name in
                reference_plan["implementation"].keys() | control_plan["implementation"].keys()
                if reference_plan["implementation"].get(name) != control_plan["implementation"].get(name)),
            "claim": "paired_fit_control_not_a_proof_of_source_meaning_or_general_transfer"}


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


def verify_source_control_supervision(plan, supervision, items, tokenizer):
    """Rebuild fit erasure and intact calibration from the bound source corpus."""
    from core.learning.semantic_candidate_contrasts import source_program_contrasts
    from core.learning.semantic_native_codec import native_sequence_for_encoding, register_encoding_from_plan
    from core.learning.semantic_native_program import source_text_from_tokens
    from core.learning.semantic_native_source_control import (
        SOURCE_ERASURE_CONTRACT,
        erase_native_source_tokens,
        source_control_mode_from_plan,
    )
    from tools.evaluate_semantic_native_checkpoint import digest

    mode = source_control_mode_from_plan(plan)
    if plan["schema"] == "aura.semantic_native_fit_plan.v3":
        from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
        from tools.train_semantic_native_program import native_grammar_supervision_sets

        fit_ids, cal_ids = set(plan["captured_fit_ids"]), set(plan["calibration_ids"])
        if (tokenizer is None or fit_ids & cal_ids or not fit_ids <= set(plan["fit_ids"])
                or not (fit_ids | cal_ids) <= set(items)
                or supervision.get("grammar_choice_contract") != GRAMMAR_CHOICE_CONTRACT):
            raise ValueError("native grammar supervision scope or independent tokenizer differs")
        controlled = mode == "source_token_erasure"
        expected_control = {**SOURCE_ERASURE_CONTRACT, "erased_fit_ids": sorted(fit_ids),
                            "unchanged_calibration_ids": sorted(cal_ids)}
        if ((controlled and supervision.get("source_evidence_control") != expected_control)
                or (not controlled and "source_evidence_control" in supervision)):
            raise ValueError("native grammar source control scope differs")
        texts = {identity: source_text_from_tokens(items[identity], tokenizer)
                 for identity in fit_ids | cal_ids}
        _sequences, groups, expected = native_grammar_supervision_sets(
            items, texts, tokenizer, tuple(sorted(fit_ids | cal_ids)),
            max_tokens=plan["max_sequence_tokens"], register_encoding=register_encoding_from_plan(plan),
            source_erasure_ids=tuple(sorted(fit_ids)) if controlled else ())
        if digest(supervision["rows"]) != digest(expected):
            raise ValueError("native grammar decisions or source tokens differ from reconstruction")
        return {"mode": mode, "source_control_verified": controlled,
                "grammar_choices_verified": True, "supervision_sequences_verified": len(expected),
                "supervised_decisions_verified": sum(len(value) for value in groups.values()),
                "erased_fit_population": len(fit_ids) if controlled else 0,
                "intact_calibration_population": len(cal_ids)}
    if mode == "source_text":
        if ("source_evidence_control" in supervision
                or any("source_control_receipt" in row for row in supervision["rows"])):
            raise ValueError("legacy native supervision cannot silently erase source evidence")
        return {"mode": mode, "source_control_verified": False}
    fit_ids, cal_ids = set(plan["captured_fit_ids"]), set(plan["calibration_ids"])
    if (tokenizer is None or fit_ids & cal_ids or not fit_ids <= set(plan["fit_ids"])
            or not (fit_ids | cal_ids) <= set(items)
            or supervision.get("source_evidence_control") != {
                **SOURCE_ERASURE_CONTRACT, "erased_fit_ids": sorted(fit_ids),
                "unchanged_calibration_ids": sorted(cal_ids)}):
        raise ValueError("native source control scope or independent tokenizer differs")
    peer_by_sha = {items[key].ir.to_program().sha(): items[key].ir.to_program()
                   for key in plan["fit_ids"]}
    peer_shas = plan["supervision_peer_program_sha256s"]
    if (len(set(peer_shas)) != len(peer_shas) or not set(peer_shas) <= set(peer_by_sha)):
        raise ValueError("native source control peer programs differ from the fit corpus")
    peers = tuple(peer_by_sha[key] for key in peer_shas)
    expected = []
    for identity in sorted(fit_ids | cal_ids):
        item = items[identity]
        if item.split != "train" or item.ir.source_text_sha256 != identity:
            raise ValueError("native source control consumes an invalid source partition")
        source, target = source_text_from_tokens(item, tokenizer), item.ir.to_program()
        programs = (target,) if plan["objective"] == "token" else source_program_contrasts(
            target, item.public_inputs, peers, source_sha256=identity, limit=plan["contrast_limit"])
        for program in sorted(programs, key=lambda value: value.sha()):
            sequence = native_sequence_for_encoding(source, program, tokenizer,
                max_tokens=plan["max_sequence_tokens"], register_encoding=register_encoding_from_plan(plan))
            receipt = None
            if identity in fit_ids:
                sequence, receipt = erase_native_source_tokens(sequence, source, tokenizer)
            expected.append({"source": identity, "program_sha256": program.sha(),
                "tokens": sequence.tokens, "continuation_start": sequence.continuation_start,
                "semantic_positions": sequence.semantic_positions, "source_control_receipt": receipt})
    if digest(supervision["rows"]) != digest(expected):
        raise ValueError("native source control tokens or receipts differ from source reconstruction")
    return {"mode": mode, "source_control_verified": True,
            "erased_fit_population": len(fit_ids), "intact_calibration_population": len(cal_ids),
            "supervision_sequences_verified": len(expected),
            "retained_nuisances": SOURCE_ERASURE_CONTRACT["retained_nuisances"]}


def verify_fit(directory, bank_directory, items, *, tokenizer=None):
    from tools.evaluate_semantic_candidate_ranker import _read_bank
    from tools.evaluate_semantic_native_checkpoint import selected_checkpoint, verified_document, verify_replay_row
    from tools.train_nested_semantic_ranker import _verified_pair
    from core.learning.semantic_native_source_control import source_control_mode_from_plan

    plan, selected = selected_checkpoint(directory)
    report = verified_document(directory / "report.json")
    bank_plan, bank_report = _verified_pair(bank_directory)
    mode = source_control_mode_from_plan(plan)
    expected_schema = ("aura.semantic_native_fit.v3" if plan["schema"] == "aura.semantic_native_fit_plan.v3"
                       else "aura.semantic_native_fit.v2" if mode == "source_token_erasure"
                       else "aura.semantic_native_fit.v1")
    if (report.get("schema") != expected_schema
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
    source_control = verify_source_control_supervision(plan, supervision, items, tokenizer)
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
            "training_source_evidence": source_control,
            "current_implementation_drift": differences, "artifacts_verified": True,
            "general_transfer_proven": False, "broad_gain_proven": False, "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "bank", "parent", "source-report", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--reference-directory", type=Path,
                        help="independently regrade an intact-source fit for paired erasure comparison")
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
    from core.learning.semantic_native_source_control import source_control_mode_from_plan
    plan = verified_document(args.directory / "plan.json", "plan_sha256")
    tokenizer = None
    if (source_control_mode_from_plan(plan) == "source_token_erasure"
            or plan["schema"] == "aura.semantic_native_fit_plan.v3"):
        from mlx_lm.utils import load_tokenizer
        tokenizer = load_tokenizer(Path(plan["model_path"]))
    result = verify_fit(args.directory, args.bank, items, tokenizer=tokenizer)
    version = "v1"
    if args.reference_directory is not None:
        reference = verify_fit(args.reference_directory, args.bank, items)
        reference_plan = verified_document(args.reference_directory / "plan.json", "plan_sha256")
        reference_report = verified_document(args.reference_directory / "report.json")
        control_report = verified_document(args.directory / "report.json")
        if result["training_source_evidence"].get("source_control_verified") is not True:
            raise ValueError("native erasure comparison lacks reconstructed source control")
        result["reference_verification"] = reference
        result["matched_source_control"] = compare_source_erasure_outcomes(
            reference_plan, plan, reference_report, control_report)
        version = "v2"
    body = {"schema": f"aura.semantic_native_fit_verification.{version}", **result}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    verified_document(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
