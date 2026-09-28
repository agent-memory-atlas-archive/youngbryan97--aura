#!/usr/bin/env python3
"""Regrade a complete native fit from source features in a separate CPU process."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def audit_grammar_identifiability(rows):
    """Find contradictory teacher labels for identical causal scoring inputs.

    This is a necessary-condition audit, not a proof that the model can learn
    the task. Source identities and target-program metadata are deliberately
    absent from the scoring fingerprint. Unscored future tokens cannot make
    an otherwise identical causal decision distinguishable.
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for row in rows:
        if (not isinstance(row.get("source"), str) or not row["source"]
                or type(row.get("decision_index")) is not int or row["decision_index"] < 0
                or type(row.get("choice_index")) is not int or row["choice_index"] < 0):
            raise ValueError("grammar identifiability needs explicit decision identities")
        groups[row["source"], row["decision_index"]].append(row)
    if not groups:
        raise ValueError("grammar identifiability needs measured supervision")
    fingerprints, kinds = defaultdict(list), Counter()
    deterministic = 0
    for identity, choices in sorted(groups.items()):
        choices.sort(key=lambda row: row["choice_index"])
        correct, kind = choices[0].get("correct_index"), choices[0].get("kind")
        if (type(correct) is not int or not 0 <= correct < len(choices)
                or kind not in {"operation", "reference", "termination"}
                or [row["choice_index"] for row in choices] != list(range(len(choices)))
                or any(row.get("correct_index") != correct or type(row.get("correct_index")) is not int
                       or row.get("kind") != kind for row in choices)):
            raise ValueError("grammar identifiability decision alternatives differ")
        inputs = []
        for row in choices:
            tokens, positions, start = (row.get("tokens"), row.get("semantic_positions"),
                                        row.get("continuation_start"))
            if (not isinstance(tokens, list) or not tokens
                    or any(type(token) is not int or token < 0 for token in tokens)
                    or type(start) is not int or not 1 <= start < len(tokens)
                    or not isinstance(positions, list) or not positions
                    or any(type(index) is not int or not start <= index < len(tokens)
                           for index in positions)
                    or sorted(set(positions)) != positions):
                raise ValueError("grammar identifiability lost causal scoring positions")
            inputs.append((tokens[:positions[-1] + 1], positions))
        key = hashlib.sha256(json.dumps(inputs, separators=(",", ":")).encode()).hexdigest()
        fingerprints[key].append({"source": identity[0], "decision_index": identity[1],
                                   "correct_index": correct})
        kinds[kind] += 1
        deterministic += len(choices) == 1
    conflicts, maximum_correct = [], 0
    for key, observations in sorted(fingerprints.items()):
        labels = Counter(row["correct_index"] for row in observations)
        maximum_correct += max(labels.values())
        if len(labels) > 1:
            conflicts.append({"scoring_input_sha256": key, "observations": observations})
    return {"schema": "aura.native_grammar_identifiability.v1", "decision_groups": len(groups),
            "unique_scoring_inputs": len(fingerprints), "by_kind": dict(sorted(kinds.items())),
            "deterministic_decisions": deterministic,
            "duplicate_scoring_input_groups": sum(len(values) > 1 for values in fingerprints.values()),
            "contradictory_scoring_input_groups": len(conflicts), "conflicts": conflicts,
            "maximum_exact_teacher_decisions": maximum_correct,
            "basis": "ordered_causal_token_choices_and_scored_target_positions",
            "learnability_proven": False, "semantic_transfer_proven": False}


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
    from core.learning.semantic_graph_counterexamples import (
        ProgramObservationCache,
        compare_program_meanings,
        counterfactual_inputs,
    )
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
    from core.learning.semantic_native_codec import (
        native_sequence_for_encoding,
        register_encoding_from_plan,
    )
    from core.learning.semantic_native_program import source_text_from_tokens
    from core.learning.semantic_native_source_control import (
        SOURCE_ERASURE_CONTRACT,
        erase_native_source_tokens,
        source_control_mode_from_plan,
    )
    from tools.evaluate_semantic_native_checkpoint import digest

    mode = source_control_mode_from_plan(plan)
    if plan["schema"] in {"aura.semantic_native_fit_plan.v3", "aura.semantic_native_fit_plan.v4",
                          "aura.semantic_native_fit_plan.v5", "aura.semantic_native_fit_plan.v6",
                          "aura.semantic_native_fit_plan.v7"}:
        from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
        from tools.train_semantic_native_program import native_grammar_supervision_sets

        choice_contract = GRAMMAR_CHOICE_CONTRACT
        if plan["schema"] in {"aura.semantic_native_fit_plan.v5", "aura.semantic_native_fit_plan.v6",
                              "aura.semantic_native_fit_plan.v7"}:
            from core.learning.semantic_native_path_objective import path_choice_contract

            choice_contract = path_choice_contract()

        fit_ids, cal_ids = set(plan["captured_fit_ids"]), set(plan["calibration_ids"])
        if (tokenizer is None or fit_ids & cal_ids or not fit_ids <= set(plan["fit_ids"])
                or not (fit_ids | cal_ids) <= set(items)
                or supervision.get("grammar_choice_contract") != choice_contract):
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
        graph_count = 0
        if plan["schema"] == "aura.semantic_native_fit_plan.v7":
            from core.learning.semantic_native_path_objective import JOINT_GRAPH_CONTRAST_CONTRACT
            from tools.train_semantic_native_program import native_supervision_sets

            peer_by_sha = {items[key].ir.to_program().sha(): items[key].ir.to_program()
                           for key in plan["fit_ids"]}
            peers = tuple(peer_by_sha[key] for key in sorted(peer_by_sha))
            graph_sequences, graph_groups = native_supervision_sets(
                items, texts, tokenizer, tuple(sorted(fit_ids | cal_ids)), peers=peers,
                contrast_limit=plan["joint_graph_contrast_limit"],
                max_tokens=plan["max_sequence_tokens"],
                register_encoding=register_encoding_from_plan(plan))
            expected_graphs = [
                {"source": identity, "choice_index": index,
                 "program_sha256": key[1], "positive": index == 0,
                 "tokens": graph_sequences[key].tokens,
                 "continuation_start": graph_sequences[key].continuation_start,
                 "semantic_positions": graph_sequences[key].semantic_positions}
                for identity, keys in sorted(graph_groups.items())
                for index, key in enumerate(keys)]
            if (supervision.get("graph_contrast_contract") != JOINT_GRAPH_CONTRAST_CONTRACT
                    or digest(supervision.get("graph_rows")) != digest(expected_graphs)):
                raise ValueError("native whole-graph supervision differs from source reconstruction")
            graph_count = len(expected_graphs)
        elif "graph_rows" in supervision or "graph_contrast_contract" in supervision:
            raise ValueError("historical native supervision acquired whole-graph rows")
        return {"mode": mode, "source_control_verified": controlled,
                "grammar_choices_verified": True,
                "supervision_sequences_verified": len(expected) + graph_count,
                "whole_graph_sequences_verified": graph_count,
                "supervised_decisions_verified": sum(len(value) for value in groups.values()),
                "identifiability": audit_grammar_identifiability(supervision["rows"]),
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


def verify_state_storage(directory, plan, report, supervision):
    """Check durable shard coverage and bytes without claiming a model rerun."""
    from core.runtime.file_read_gateway import open_stable_readonly_binary
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document

    contract = plan.get("prefix_storage_contract")
    receipt = report.get("prefix_storage_receipt")
    reuse = plan.get("reused_prefix_contract")
    if reuse is not None:
        from tools.semantic_native_prefix_reuse import (
            prefix_reuse_contract,
            source_capture_receipts,
        )

        origin = Path(reuse["source_directory"])
        if (report.get("reused_prefix_contract") != reuse
                or prefix_reuse_contract(origin, plan) != reuse
                or digest(supervision["rows"]) != reuse["source_supervision_rows_sha256"]):
            raise ValueError("native frozen prefix reuse differs")
        expected_captures = source_capture_receipts(
            origin, sources=sorted({row["source"] for row in supervision["rows"]}),
            plan_sha256=reuse["source_plan_sha256"])
        if (digest(expected_captures) != reuse["capture_inventory_sha256"]
                or report.get("prefix_capture_receipts") != expected_captures):
            raise ValueError("native reused capture inventory differs")
        shard_plan_sha = reuse["source_plan_sha256"]
        shard_directory = origin
    else:
        if "reused_prefix_contract" in report:
            raise ValueError("undeclared native frozen prefix reuse")
        shard_plan_sha = plan["plan_sha256"]
        shard_directory = directory
    if contract is None:
        if receipt is not None:
            raise ValueError("undeclared frozen state storage")
        return None
    bound = contract.get("max_resident_bytes")
    if (contract != {"schema": "aura.frozen_state_storage_contract.v1", "mode": "source_shards",
                    "max_resident_bytes": bound, "lossy_compression": False,
                    "all_alternatives_retained": True}
            or type(bound) is not int or not 0 < bound <= 4096 * 1024 * 1024
            or not isinstance(receipt, dict)
            or receipt.get("schema") != "aura.frozen_state_store.v1"
            or receipt.get("plan_sha256") != shard_plan_sha
            or receipt.get("max_resident_bytes") != bound
            or receipt.get("lossy_compression") is not False
            or type(receipt.get("peak_resident_bytes")) is not int
            or not 0 <= receipt["peak_resident_bytes"] <= bound):
        raise ValueError("frozen state storage contract differs")
    expected = {(row["source"], row["decision_index"], row["choice_index"]): digest(row["tokens"])
                for row in supervision["rows"]}
    expected.update({(row["source"], -1, row["choice_index"]): digest(row["tokens"])
                     for row in supervision.get("graph_rows", ())})
    observed, sources, total_bytes = {}, set(), 0
    for shard in receipt["shards"]:
        source = shard["source"]
        name = hashlib.sha256(source.encode()).hexdigest()
        if (source in sources or shard["plan_sha256"] != shard_plan_sha
                or shard["schema"] != "aura.frozen_state_shard.v1"
                or type(shard["array_bytes"]) is not int or not 0 < shard["array_bytes"] <= bound
                or type(shard["file_bytes"]) is not int
                or not shard["array_bytes"] <= shard["file_bytes"] <= shard["array_bytes"] + 1024 * 1024):
            raise ValueError("frozen state source shard differs")
        sources.add(source)
        if verified_document(shard_directory / "prefix-states" / f"{name}.json") != shard:
            raise ValueError("frozen state manifest differs from report")
        with open_stable_readonly_binary(shard_directory / "prefix-states" / f"{name}.safetensors",
                                        max_bytes=shard["file_bytes"]) as (handle, identity):
            content = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                content.update(chunk)
            if identity.size != shard["file_bytes"] or content.hexdigest() != shard["weights_sha256"]:
                raise ValueError("frozen state durable bytes differ")
        for index, row in enumerate(shard["arrays"]):
            key = tuple(row["key"])
            if key in observed or key[0] != source or row["tensor"] != str(index):
                raise ValueError("frozen state alternative repeats or changes source")
            observed[key] = row["sequence_sha256"]
        total_bytes += shard["file_bytes"]
    if (observed != expected or receipt["sequences"] != len(expected)
            or receipt["sources"] != len(sources)):
        raise ValueError("frozen state sequence coverage differs")
    return {"sources": len(sources), "sequences": len(expected), "file_bytes": total_bytes,
            "durable_bytes_and_sequence_coverage_verified": True,
            "hidden_states_independently_recomputed": False}


def persisted_source_pairs(pairs):
    """Compare reconstructed witnesses in the signed plan's JSON data model."""
    return json.loads(json.dumps(pairs, sort_keys=True, allow_nan=False))


def verify_fit(directory, bank_directory, items, *, tokenizer=None):
    from core.learning.semantic_native_source_control import source_control_mode_from_plan
    from tools.evaluate_semantic_candidate_ranker import _read_bank
    from tools.evaluate_semantic_native_checkpoint import (
        selected_checkpoint,
        verified_document,
        verify_replay_row,
    )
    from tools.train_nested_semantic_ranker import _verified_pair

    plan, selected = selected_checkpoint(directory)
    report = verified_document(directory / "report.json")
    from tools.semantic_native_execution import execution_from_plan
    execution = execution_from_plan(plan)
    if execution is not None and report.get("execution_contract") != execution:
        raise ValueError("native fit report arithmetic differs from its plan")
    bank_plan, bank_report = _verified_pair(bank_directory)
    mode = source_control_mode_from_plan(plan)
    expected_schema = ("aura.semantic_native_fit.v7" if plan["schema"] == "aura.semantic_native_fit_plan.v7"
                       else "aura.semantic_native_fit.v6" if plan["schema"] == "aura.semantic_native_fit_plan.v6"
                       else "aura.semantic_native_fit.v5" if plan["schema"] == "aura.semantic_native_fit_plan.v5"
                       else "aura.semantic_native_fit.v4" if plan["schema"] == "aura.semantic_native_fit_plan.v4"
                       else "aura.semantic_native_fit.v3" if plan["schema"] == "aura.semantic_native_fit_plan.v3"
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
    if plan.get("objective") == "grammar_source_pairs":
        from core.learning.semantic_native_source_pairs import native_source_pair_plan

        pair_planner = native_source_pair_plan
        if plan["schema"] == "aura.semantic_native_fit_plan.v6" or (
                plan["schema"] == "aura.semantic_native_fit_plan.v7"
                and "grammar_source_pair_inventory" in plan):
            from core.learning.semantic_native_typed_source_pairs import (
                native_typed_source_pair_plan,
                typed_source_pair_inventory,
            )

            pair_planner = native_typed_source_pair_plan
        pairs = pair_planner(
            tuple(items[identity] for identity in plan["fit_ids"]), plan["fit_ids"],
            register_encoding=plan["register_encoding"])
        if (persisted_source_pairs(pairs) != plan["grammar_source_pair_fit_partners"]
                or plan["grammar_source_pair_updates"] != sum(
                    identity in pairs for identity in plan["scheduled_fit_ids"])):
            raise ValueError("native fit source-pair supervision differs")
        if (plan["schema"] in {"aura.semantic_native_fit_plan.v6", "aura.semantic_native_fit_plan.v7"}
                and "grammar_source_pair_inventory" in plan
                and plan["grammar_source_pair_inventory"]
                    != typed_source_pair_inventory(pairs, plan["scheduled_fit_ids"])):
            raise ValueError("native fit typed source-pair inventory differs")
    checkpoints = [verified_document(directory / f"checkpoint-{row['step']}.json") for row in report["checkpoints"]]
    if checkpoints != report["checkpoints"] or selected not in checkpoints:
        raise ValueError("native fit checkpoint report differs")
    supervision = verified_document(directory / "supervision.json")
    if (supervision["plan_sha256"] != plan["plan_sha256"]
            or supervision["receipt_sha256"] != report["supervision_receipt_sha256"]
            or len(supervision["rows"]) + len(supervision.get("graph_rows", ()))
                != report["prefix_sequence_population"]
            or {row["source"] for row in supervision["rows"]} != set(plan["captured_fit_ids"]) | set(plan["calibration_ids"])
            or report["gradient_source_population"] != len(set(plan["scheduled_fit_ids"]))):
        raise ValueError("native fit supervision coverage differs")
    source_control = verify_source_control_supervision(plan, supervision, items, tokenizer)
    storage = verify_state_storage(directory, plan, report, supervision)
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
            "frozen_state_storage": storage,
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
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import (
        configure_refit_environment,
        load_source_examples,
        source_bundle_arguments,
    )
    from tools.train_nested_semantic_ranker import _verified_pair

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
            or plan["schema"] in {"aura.semantic_native_fit_plan.v3", "aura.semantic_native_fit_plan.v4",
                                  "aura.semantic_native_fit_plan.v5", "aura.semantic_native_fit_plan.v6",
                                  "aura.semantic_native_fit_plan.v7"}):
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
