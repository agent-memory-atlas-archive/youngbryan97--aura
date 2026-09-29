#!/usr/bin/env python3
"""Regrade a target-blind native decode from its durable source and programs."""

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

from core.verify.invariants import Violation, invariant  # noqa: E402

_INTERVENTION_DATASETS = frozenset({
    "operation_intervention", "definition_intervention", "equation_intervention",
    "role_intervention", "dependency_intervention",
})


def verify_grammar_row(row, *, example, identity, plan_sha256):
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent

    if (row["source_sha256"] != identity or row["plan_sha256"] != plan_sha256
            or row["construction"] != example.construction_id
            or row["target_available_to_scorer"] is not False):
        raise ValueError("native grammar row source or target-blind authority differs")
    if (type(row["bound_forced_completion"]) is not bool
            or type(row["depth_bound_reached"]) is not bool
            or row["depth_bound_reached"] is not (
                row["bound_forced_completion"]
                or row["decode_status"] == "disconnected_at_depth_bound")):
        raise ValueError("native grammar row depth-bound accounting differs")
    payload = row["program"]
    program = None
    if payload is not None:
        instructions = tuple(Instruction(op, tuple(arguments))
                             for op, arguments in payload["instructions"])
        program = Program(len(example.inputs), instructions)
        if program.to_dict() != payload:
            raise ValueError("native grammar program serialization differs")
    completed = row["decode_status"] == "completed"
    if completed and program is None:
        raise ValueError("native grammar completion has no program")
    equivalent = completed and semantic_programs_structurally_equivalent(program, example.program)
    answer_correct = False
    if completed:
        try:
            answer_correct = program.run(example.inputs) == example.program.run(example.inputs)
        except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
            pass
    if (row["program_equivalent"] is not equivalent
            or row["answer_correct"] is not answer_correct
            or (not completed and (equivalent or answer_correct))):
        raise ValueError("native grammar row outcome differs from independent execution")
    return equivalent, answer_correct


def verified_source_evidence(plan, report, rows):
    """Preserve the source intervention across plan, report, and every row."""
    mode = plan.get("source_evidence", "source_text")
    if (mode not in {"source_text", "source_token_erasure", "source_pair_swap"}
            or report.get("source_evidence", "source_text") != mode
            or any(row.get("source_evidence", "source_text") != mode for row in rows)):
        raise ValueError("native grammar source-evidence mode differs")
    return mode


def replay_greedy_decisions(row, *, example, plan, tokenizer=None, max_sequence_tokens=None,
                            scored_source=None):
    from core.learning.semantic_native_grammar import (
        NativeGrammarIncompleteError,
        decode_native_grammar,
    )

    trace = row["decision_trace"]
    if not isinstance(trace, list):
        raise ValueError("native grammar decision trace is missing")
    parameter_scales = row.get("decision_parameter_scales")
    if plan["schema"].endswith(".v10"):
        if not isinstance(parameter_scales, list) or len(parameter_scales) != len(trace):
            raise ValueError("native factorized parameter receipts are incomplete")
    elif parameter_scales is not None:
        raise ValueError("historical native grammar acquired factorized parameters")
    verify_inputs = "source_evidence" in plan
    scored_source = example.source_text if scored_source is None else scored_source
    input_receipts = row.get("score_input_receipts")
    if verify_inputs and (tokenizer is None or max_sequence_tokens is None
                          or not isinstance(input_receipts, list)
                          or len(input_receipts) != len(trace)):
        raise ValueError("native grammar scored-input receipts are missing")
    consumed = 0
    anchor_sha256 = None

    def recorded_scores(choices):
        nonlocal consumed, anchor_sha256
        if consumed >= len(trace):
            raise ValueError("native grammar decision trace ended before the graph")
        entry = trace[consumed]
        if parameter_scales is not None:
            verify_factorized_parameter_receipt(plan, entry, choices, parameter_scales[consumed])
        if verify_inputs:
            from core.learning.semantic_native_program import native_text_decision_sequence
            from core.learning.semantic_native_source_control import (
                apply_native_source_evidence,
                native_score_input_receipt,
            )

            expected_inputs = []
            sequences = []
            for choice in choices:
                sequence = native_text_decision_sequence(
                    scored_source, choice.text, (choice.span,), tokenizer,
                    max_tokens=max_sequence_tokens)
                sequence, control = apply_native_source_evidence(
                    sequence, scored_source, tokenizer,
                    mode="source_text" if plan["source_evidence"] == "source_pair_swap"
                    else plan["source_evidence"])
                expected_inputs.append(native_score_input_receipt(sequence, control))
                sequences.append(sequence)
            if input_receipts[consumed] != expected_inputs:
                raise ValueError("native grammar scored-input tokens or source control differ")
            if plan["schema"].endswith((".v7", ".v8", ".v9", ".v10")):
                anchor_sha256 = verified_trie_anchor(
                    sequences, row["prefix_execution"], anchor_sha256)
        consumed += 1
        values = [choice.value for choice in choices]
        scores = entry["scores"]
        if (entry["choices"] != values or not isinstance(scores, list)
                or len(scores) != len(values)
                or any(type(score) not in (int, float) or not math.isfinite(score)
                       for score in scores)
                or entry["chosen"] != values[max(range(len(scores)), key=scores.__getitem__)]):
            raise ValueError("native grammar decision choices or winning score differ")
        return tuple(scores)

    types = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                  for value in verified_public_inputs(example))
    try:
        generated = decode_native_grammar(types, recorded_scores,
                                          max_steps=plan["max_steps"],
                                          register_encoding=plan["register_encoding"])
    except NativeGrammarIncompleteError as failure:
        program, replayed_trace, forced = failure.program, failure.trace, False
        status = "disconnected_at_depth_bound"
    else:
        program, replayed_trace = generated.program, generated.trace
        forced, status = generated.bound_forced_completion, "completed"
    if (consumed != len(trace) or json.loads(json.dumps(replayed_trace)) != trace
            or status != row["decode_status"]
            or program.to_dict() != row["program"]
            or forced is not row["bound_forced_completion"]):
        raise ValueError("native grammar decision replay differs from saved graph")
    if plan["schema"].endswith((".v7", ".v8", ".v9", ".v10")) and anchor_sha256 is None:
        raise ValueError("native grammar trie has no scored source anchor")


def replay_search_decisions(row, *, example, plan, input_types,
                            input_receipts_for_choices, input_receipt_for_program):
    """Rebuild every searched branch from saved scores, including dead ends."""
    from core.learning.semantic_native_search import search_native_grammar
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent

    observed = row.get("search")
    transcript = observed.get("score_transcript") if isinstance(observed, dict) else None
    receipts = row.get("score_input_receipts")
    if (not isinstance(transcript, list) or not transcript
            or not isinstance(receipts, list) or len(receipts) != len(transcript)):
        raise ValueError("native search lacks a complete decision-score transcript")
    consumed = 0

    def recorded_scores(choices):
        nonlocal consumed
        if consumed >= len(transcript):
            raise ValueError("native search transcript ended before exploration")
        entry = transcript[consumed]
        values = [choice.value for choice in choices]
        scores = entry.get("scores") if isinstance(entry, dict) else None
        if (not isinstance(entry, dict) or entry.get("choices") != values
                or not isinstance(scores, list)
                or len(scores) != len(choices)
                or any(type(score) not in {int, float} or not math.isfinite(score)
                       for score in scores)
                or receipts[consumed] != input_receipts_for_choices(choices)):
            raise ValueError("native search scored choices or source inputs differ")
        consumed += 1
        return tuple(scores)

    replayed = search_native_grammar(input_types, recorded_scores,
        max_steps=plan["max_steps"], max_nodes=plan["search_nodes"],
        completions=plan["search_completions"],
        register_encoding=plan["register_encoding"],
        score_mode=plan["search_score_mode"])
    if consumed != len(transcript):
        raise ValueError("native search transcript contains unvisited decisions")
    graph_scores = observed.get("complete_graph_scores")
    graph_receipts = observed.get("graph_score_input_receipts")
    if (not isinstance(graph_scores, list) or len(graph_scores) != len(replayed.candidates)
            or any(type(score) not in {int, float} or not math.isfinite(score)
                   for score in graph_scores)
            or not isinstance(graph_receipts, list)
            or graph_receipts != [input_receipt_for_program(candidate.result.program)
                                  for candidate in replayed.candidates]):
        raise ValueError("native search lacks source-bound whole-graph scores")
    chosen = max(range(len(graph_scores)), key=graph_scores.__getitem__) if graph_scores else None
    expected = {"expanded_nodes": replayed.expanded_nodes,
        "scored_decisions": replayed.scored_decisions,
        "scored_alternatives": replayed.scored_alternatives,
        "disconnected_leaves": replayed.disconnected_leaves,
        "pruned_prefixes": replayed.pruned_prefixes,
        "frontier_nodes": replayed.frontier_nodes,
        "frontier_log_probability_bound": replayed.frontier_log_probability_bound,
        "halt_reason": replayed.halt_reason,
        "requested_top_k_proven": replayed.requested_top_k_proven,
        "score_transcript": transcript,
        "graph_score_input_receipts": graph_receipts,
        "selected_index": chosen, "complete_graph_scores": graph_scores,
        "proposals": [{"program": candidate.result.program.to_dict(),
            "log_probability": candidate.log_probability,
            "decision_trace": candidate.result.trace,
            "bound_forced_completion": candidate.result.bound_forced_completion}
            for candidate in replayed.candidates],
        "observed_program_reach": any(semantic_programs_structurally_equivalent(
            candidate.result.program, example.program) for candidate in replayed.candidates)}
    if observed != json.loads(json.dumps(expected)):
        raise ValueError("native search result differs from decision replay")
    selected = None if chosen is None else replayed.candidates[chosen].result
    if (row["decode_status"] != ("search_without_completion" if selected is None else "completed")
            or row["program"] != (None if selected is None else selected.program.to_dict())
            or row["decision_trace"] != json.loads(json.dumps(
                () if selected is None else selected.trace))
            or row["bound_forced_completion"] is not (
                False if selected is None else selected.bound_forced_completion)):
        raise ValueError("native search selection differs from saved graph")


def verify_factorized_parameter_receipt(plan, entry, choices, receipt):
    from core.learning.semantic_native_factorized_residual import (
        native_competition_kind,
        validated_kind_scales,
    )

    scales = validated_kind_scales(plan["factorized_residual"]["selected_scales"])
    kind = native_competition_kind(choices)
    if (receipt.get("kind") != kind or entry.get("kind") != kind
            or type(receipt.get("scale")) not in {int, float} or receipt["scale"] != scales[kind]
            or type(receipt.get("adapter_sites")) is not int or receipt["adapter_sites"] < 1):
        raise ValueError("native factorized parameter receipt differs from typed selection")


def source_separation_summary(source_examples, target_examples):
    source_texts = {hashlib.sha256(item.source_text.encode()).hexdigest()
                    for item in source_examples}
    target_texts = {hashlib.sha256(item.source_text.encode()).hexdigest()
                    for item in target_examples}
    source_constructions = {item.construction_id for item in source_examples}
    target_constructions = {item.construction_id for item in target_examples}
    source_topologies = {item.topology_id for item in source_examples}
    target_topologies = {item.topology_id for item in target_examples}
    if not source_texts or not target_texts or source_texts & target_texts:
        raise ValueError("native grammar source text overlaps the training cohort")
    if source_constructions & target_constructions:
        raise ValueError("native grammar construction overlaps the training cohort")
    return {"source_examples": len(source_examples),
            "source_constructions": len(source_constructions),
            "target_constructions": len(target_constructions),
            "shared_topologies": sorted(source_topologies & target_topologies),
            "source_text_overlap": 0, "construction_overlap": 0}


def verify_source_separation(training, source_report_path, bundles, target_examples):
    from core.learning.semantic_program_campaign import _sha
    from core.learning.semantic_program_feature_materialization import (
        rebuild_semantic_feature_selection,
    )
    from tools.refit_semantic_argument_proposals import source_bundle_arguments

    raw = source_report_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != training["source_report_sha256"]:
        raise ValueError("native grammar source report differs from fitted checkpoint")
    report = json.loads(raw)
    if (report.get("report_sha256") != _sha({key: value for key, value in report.items()
                                             if key != "report_sha256"})
            or report.get("fit_complete") is not True):
        raise ValueError("native grammar source report identity differs")
    paths = source_bundle_arguments(report, bundles=bundles)
    expected = report["representation_compatibility"]["source_feature_manifest_sha256s"]
    source_examples = []
    for value in paths:
        name, _, path = value.partition("=")
        manifest = json.loads((Path(path) / "manifest.json").read_text())
        if manifest.get("manifest_sha256") != expected[name]:
            raise ValueError(f"native grammar source manifest differs: {name}")
        _, examples = rebuild_semantic_feature_selection(manifest)
        source_examples.extend(examples)
    return {"source_report_sha256": training["source_report_sha256"],
            "source_manifests": expected,
            **source_separation_summary(source_examples, target_examples)}


def verified_weight_mode(plan, report):
    version = plan.get("schema", "").rsplit(".", 1)[-1]
    if (version not in {"v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"}
            or plan["schema"] != f"aura.semantic_native_grammar_plan.{version}"
            or report.get("schema") != f"aura.semantic_native_grammar.{version}"):
        raise ValueError("native grammar schema versions differ")
    if version == "v1":
        if "weight_mode" in plan or "weight_mode" in report:
            raise ValueError("historical native grammar weight mode was fitted")
        return "fitted"
    modes = ({"factorized"} if version == "v10" else {"residual"} if version == "v9"
             else {"fitted", "base"})
    if (plan.get("weight_mode") not in modes
            or report.get("weight_mode") != plan["weight_mode"]):
        raise ValueError("native grammar weight mode differs")
    return plan["weight_mode"]


def verified_input_grounding(plan, report):
    version = plan.get("schema", "").rsplit(".", 1)[-1]
    expected = ("semantic_public_character_inputs.v1" if version in {"v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"}
                else "declared_public_inputs")
    if (plan.get("input_grounding") != expected
            or report.get("input_grounding") != plan["input_grounding"]):
        raise ValueError("native grammar input grounding differs")
    return plan["input_grounding"]


def verified_dataset(plan, report):
    version = plan.get("schema", "").rsplit(".", 1)[-1]
    verified_source_window(plan, report)
    if version in {"v1", "v2"}:
        if any(name in document for document in (plan, report)
               for name in ("dataset", "seed")):
            raise ValueError("historical native grammar dataset differed")
        return "natural_request", 3141592
    dataset, seed = plan.get("dataset"), plan.get("seed")
    allowed = ({"definition_intervention", "equation_intervention"} if version == "v4"
               else {"natural_request", "operation_intervention"})
    if version == "v5":
        allowed = {"role_intervention", "dependency_intervention"}
    if version == "v11":
        allowed = {"relation_transfer_controls"}
    if version == "v7":
        allowed = {"operation_intervention", "definition_intervention", "equation_intervention",
                   "role_intervention", "dependency_intervention"}
    if version in {"v6", "v8", "v9", "v10", "v12"}:
        allowed = {"retained_validation", "retained_test"}
        if (seed != 0 or not isinstance(plan.get("source_cohort_basis"), dict)
                or plan["source_cohort_basis"] != report.get("source_cohort_basis")):
            raise ValueError("retained native development source basis differs")
    elif "source_cohort_basis" in plan or "source_cohort_basis" in report:
        raise ValueError("historical native grammar cannot acquire a retained source basis")
    if (dataset not in allowed
            or type(seed) is not int or seed < 0
            or report.get("dataset") != dataset or report.get("seed") != seed):
        raise ValueError("native grammar dataset or seed differs")
    return dataset, seed


def verified_source_window(plan, report):
    """A window must retain its complete population identity in both artifacts."""
    window = plan.get("source_window")
    if not plan["schema"].endswith(".v12"):
        if window is not None or "source_window" in report:
            raise ValueError("historical native grammar acquired a source window")
        return None
    if (not isinstance(window, dict) or set(window) != {
            "offset", "count", "population", "ordered_sources_sha256"}
            or report.get("source_window") != window
            or any(type(window[key]) is not int for key in ("offset", "count", "population"))
            or window["population"] != 500 or window["offset"] < 0 or window["count"] < 1
            or window["offset"] + window["count"] > 500
            or window["count"] != len(plan["sources"])
            or not isinstance(window["ordered_sources_sha256"], str)
            or len(window["ordered_sources_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in window["ordered_sources_sha256"])):
        raise ValueError("native grammar complete source window differs")
    return window


@invariant("learning.native_development_windows_bind_complete_population", scope="learning",
           owner="tools/verify_semantic_native_grammar.py", observational=False)
def _native_development_window_contract():
    plan = {"schema": "aura.semantic_native_grammar_plan.v12", "sources": ["source"],
            "source_window": {"offset": 499, "count": 1, "population": 500,
                              "ordered_sources_sha256": "a" * 64}}
    verified_source_window(plan, plan)
    for change in ({"count": True}, {"offset": 500}, {"population": 499}):
        invalid = {**plan, "source_window": {**plan["source_window"], **change}}
        try:
            verified_source_window(invalid, invalid)
        except ValueError:
            continue
        yield Violation(subject="source-window", message="invalid complete population was admitted")


def verified_prefix_execution(plan, report, row=None):
    """Require complete branch accounting only for the new reuse contract."""
    is_trie = plan["schema"].endswith((".v7", ".v8", ".v9", ".v10"))
    if is_trie:
        if plan.get("prefix_strategy") != "trie" or report.get("prefix_strategy") != "trie":
            raise ValueError("native grammar trie strategy differs")
    elif "prefix_strategy" in plan or "prefix_strategy" in report:
        raise ValueError("historical native grammar acquired a prefix strategy")
    if row is None:
        return
    receipt = row.get("prefix_execution")
    if not is_trie:
        if receipt is not None:
            raise ValueError("historical native grammar acquired a prefix receipt")
        return
    competitions = row["score_input_receipts"]
    if (not isinstance(receipt, dict) or receipt.get("schema") != "aura.frozen_prefix_branches.v2"
            or receipt.get("suffix_computation_unchanged") is not True
            or type(receipt.get("anchor_tokens")) is not int or receipt["anchor_tokens"] < 1
            or receipt.get("trie_calls") != len(competitions)
            or receipt.get("branches") != sum(len(choices) for choices in competitions)
            or not all(choices for choices in competitions)):
        raise ValueError("native grammar trie execution does not cover every choice")


def verified_trie_anchor(sequences, receipt, previous_sha256=None):
    from core.learning.frozen_prefix_branches import native_source_anchor

    anchor = native_source_anchor(sequences)
    observed = hashlib.sha256(json.dumps(anchor, separators=(",", ":"))
                              .encode("ascii")).hexdigest()
    if (receipt["anchor_tokens"] != len(anchor)
            or receipt.get("anchor_token_sha256") != observed
            or previous_sha256 not in (None, observed)):
        raise ValueError("native grammar trie source anchor differs")
    return observed


def verified_examples(plan, *, dataset, seed):
    from core.learning.semantic_program_corpus_natural import (
        build_semantic_program_natural_request_corpus,
    )

    count = len(plan["sources"])
    if dataset in {"retained_validation", "retained_test"}:
        from tools.semantic_native_retained_sources import load_retained_native_sources

        basis = plan["source_cohort_basis"]
        if basis["split"] != dataset.removeprefix("retained_"):
            raise ValueError("retained native development split differs")
        bundles = [f"{name}={path}" for name, path in sorted(basis["source_manifest_paths"].items())]
        if plan.get("schema", "").endswith(".v12"):
            from tools.semantic_native_retained_sources import retained_native_source_window

            window = verified_source_window(plan, plan)
            examples, rebuilt, rebuilt_window = retained_native_source_window(
                basis["source_report_path"], bundles, split=basis["split"],
                offset=window["offset"], count=window["count"])
            if rebuilt_window != window:
                raise ValueError("native grammar source window reconstruction differs")
        else:
            examples, rebuilt = load_retained_native_sources(basis["source_report_path"],
                bundles, split=basis["split"], count=count)
        if rebuilt != basis:
            raise ValueError("retained native development source reconstruction differs")
        return examples
    if dataset == "natural_request":
        corpus = build_semantic_program_natural_request_corpus(
            seed=seed, examples_per_schema_domain=3,
        )
        ordered = tuple(corpus[index] for sample in range(24)
                        for index in (sample, sample + 24, sample + 48))
    elif dataset == "relation_transfer_controls":
        from tools.semantic_native_relation_transfer import build_native_relation_transfer

        ordered = tuple(case.controlled for case in build_native_relation_transfer(seed=seed))
        if count != len(ordered):
            raise ValueError("native relation transfer needs its complete controlled inventory")
    else:
        from tools.semantic_native_operation_interventions import (
            build_native_operation_interventions,
        )
        if dataset == "operation_intervention":
            pairs = build_native_operation_interventions(seed=seed)
        elif dataset in {"role_intervention", "dependency_intervention"}:
            from tools.semantic_native_graph_interventions import build_native_graph_interventions

            pairs = build_native_graph_interventions(seed=seed, kind=dataset.removesuffix("_intervention"))
        else:
            from tools.semantic_native_paraphrase_interventions import (
                build_native_paraphrase_interventions,
            )

            pairs = build_native_paraphrase_interventions(
                seed=seed, style=dataset.removesuffix("_intervention"))
        groups = {}
        for pair in pairs:
            groups.setdefault(pair.original.topology_id, []).append(pair)
        ordered = tuple(example for cohort in zip(*groups.values(), strict=True)
                        for pair in cohort for example in (pair.original, pair.changed))
        if count % 2:
            raise ValueError("native grammar intervention split a source pair")
    if not 1 <= count <= len(ordered):
        raise ValueError("native grammar population exceeds the source inventory")
    return ordered[:count]


def verified_pair_totals(rows, *, dataset):
    if dataset not in _INTERVENTION_DATASETS:
        return {"pair_count": 0, "pair_exact": 0, "source_responsive": 0}
    if len(rows) % 2:
        raise ValueError("native grammar intervention rows split a source pair")
    exact = responsive = 0
    for index in range(0, len(rows), 2):
        left, right = rows[index], rows[index + 1]
        exact += (left["program_equivalent"] and left["answer_correct"]
                  and right["program_equivalent"] and right["answer_correct"])
        if left["decode_status"] == right["decode_status"] == "completed":
            left_steps = left["program"]["instructions"]
            right_steps = right["program"]["instructions"]
            if dataset in {"role_intervention", "dependency_intervention"}:
                if (len(left_steps) != len(right_steps)
                        or [step[0] for step in left_steps] != [step[0] for step in right_steps]):
                    continue
                changed_indices = [i for i in range(len(left_steps))
                                   if left_steps[i][1] != right_steps[i][1]]
                if len(changed_indices) != 1:
                    continue
                before, after = left_steps[changed_indices[0]][1], right_steps[changed_indices[0]][1]
                if dataset == "role_intervention":
                    responsive += len(before) == 2 and after == before[::-1]
                else:
                    responsive += (len(before) == len(after)
                                   and sum(before[i] != after[i] for i in range(len(before))) == 1)
            else:
                responsive += (left_steps[:-1] == right_steps[:-1]
                               and left_steps[-1][1] == right_steps[-1][1]
                               and left_steps[-1][0] != right_steps[-1][0])
    return {"pair_count": len(rows) // 2, "pair_exact": exact,
            "source_responsive": responsive}


def verified_public_inputs(example):
    from core.learning.semantic_public_inputs import semantic_public_character_inputs

    recovered = semantic_public_character_inputs(example.source_text)
    if recovered.values != example.inputs:
        raise ValueError("native grammar public source values differ from annotations")
    return recovered.values


def audit_grammar_meanings(rows, examples):
    """Keep procedure fidelity, proven denotation, and sampled agreement distinct."""
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_graph_counterexamples import (
        compare_program_meanings,
        counterfactual_inputs,
    )

    if len(rows) != len(examples) or not rows:
        raise ValueError("native meaning audit needs complete matched source rows")
    comparisons = []
    for row, example in zip(rows, examples, strict=True):
        identity = hashlib.sha256(example.source_text.encode()).hexdigest()
        structural, answer = verify_grammar_row(row, example=example, identity=identity,
            plan_sha256=row["plan_sha256"])
        if row["decode_status"] != "completed":
            meaning = {"status": "unavailable", "method": "incomplete_native_graph"}
        else:
            program = Program(len(example.inputs), tuple(Instruction(op, tuple(arguments))
                              for op, arguments in row["program"]["instructions"]))
            meaning = compare_program_meanings(example.program, program,
                counterfactual_inputs(example.inputs))
        comparisons.append({"source_sha256": identity, "procedure_equivalent": structural,
                            "observed_answer_correct": answer, "meaning": meaning})
    return {"population": len(rows),
            "procedure_equivalent": sum(row["procedure_equivalent"] for row in comparisons),
            "proven_output_and_domain_equivalent": sum(row["meaning"]["status"] == "equivalent"
                                                       for row in comparisons),
            "witnessed_different": sum(row["meaning"]["status"] == "different" for row in comparisons),
            "meaning_unknown": sum(row["meaning"]["status"] == "unknown" for row in comparisons),
            "meaning_unavailable": sum(row["meaning"]["status"] == "unavailable" for row in comparisons),
            "comparisons": comparisons,
            "claim": "grading_only_no_procedure_fidelity_or_source_understanding_waiver",
            "historical_totals_unchanged": True}


def verify_grammar(directory, training_directory):
    from tools.evaluate_semantic_native_checkpoint import selected_checkpoint, verified_document

    training, selected = selected_checkpoint(training_directory)
    plan = verified_document(directory / "plan.json", "plan_sha256")
    report = verified_document(directory / "report.json")
    weight_mode = verified_weight_mode(plan, report)
    scored_checkpoint = selected
    if weight_mode == "residual":
        from tools.verify_semantic_native_residual import verify_residual

        contract = plan.get("residual_calibration")
        if (not isinstance(contract, dict) or contract != report.get("residual_calibration")
                or contract.get("source_only") is not True
                or contract.get("serving_authority") is not False
                or contract.get("baseline_checkpoint_receipt_sha256") != selected["receipt_sha256"]):
            raise ValueError("native grammar residual calibration contract differs")
        result = verify_residual(Path(contract["directory"]), training_directory)
        if (result["report_receipt_sha256"] != contract["report_receipt_sha256"]
                or result["selected_scale"] != contract["selected_scale"]
                or result["selected_scale"] <= 0.
                or result["current_implementation_drift"]):
            raise ValueError("native grammar residual source calibration differs")
        scored_checkpoint = verified_document(training_directory /
            f"checkpoint-{result['candidate_step']}.json")
    elif "residual_calibration" in plan or "residual_calibration" in report:
        raise ValueError("historical native grammar acquired a residual calibration")
    if weight_mode == "factorized":
        from tools.verify_semantic_native_factorized_residual import verify_factorized_residual

        contract = plan.get("factorized_residual")
        if (not isinstance(contract, dict) or contract != report.get("factorized_residual")
                or contract.get("source_only") is not True
                or contract.get("serving_authority") is not False
                or contract.get("baseline_checkpoint_receipt_sha256") != selected["receipt_sha256"]):
            raise ValueError("native factorized source contract differs")
        result = verify_factorized_residual(Path(contract["report_path"]), training_directory)
        if (result["report_receipt_sha256"] != contract["report_receipt_sha256"]
                or result["selected_scales"] != contract["selected_scales"]
                or not any(result["selected_scales"].values())
                or result["current_implementation_drift"]):
            raise ValueError("native factorized source measurement differs")
        scored_checkpoint = verified_document(training_directory /
            f"checkpoint-{result['candidate_step']}.json")
    elif "factorized_residual" in plan or "factorized_residual" in report:
        raise ValueError("historical native grammar acquired factorized source selection")
    verified_prefix_execution(plan, report)
    if plan["schema"].endswith((".v7", ".v8", ".v9", ".v10")):
        from tools.semantic_native_execution import execution_from_plan

        execution = execution_from_plan(training)
        if execution is None or execution["precision"] != "float32":
            raise ValueError("native grammar trie training arithmetic differs")
    input_grounding = verified_input_grounding(plan, report)
    dataset, seed = verified_dataset(plan, report)
    if (plan["schema"].endswith((".v6", ".v8", ".v9", ".v10", ".v12"))
            and plan["source_cohort_basis"]["source_report_sha256"] != training["source_report_sha256"]):
        raise ValueError("retained native source basis differs from the fitted checkpoint")
    if (plan["training_plan_sha256"] != training["plan_sha256"]
            or plan["checkpoint_receipt_sha256"] != scored_checkpoint["receipt_sha256"]
            or plan["model_descriptor_sha256"] != training["model_descriptor_sha256"]
            or plan["pointer_sha256"] != training["pointer_sha256"]
            or plan["target_available_to_scorer"] is not False
            or report["target_available_to_scorer"] is not False
            or any(plan[key] is not False or report[key] is not False
                   for key in ("serving_authority", "qualification_evidence"))
            or plan["candidate_inventory"] != "none"
            or report["candidate_inventory"] != "none"
            or plan["search_mode"] not in {"greedy", "best_first_then_complete_graph_score"}
            or (plan["search_mode"] == "greedy") != (plan["search_completions"] == 0)
            or plan["search_score_mode"] not in {"normalized_choices", "native_nonpositive"}
            or type(plan["search_nodes"]) is not int or plan["search_nodes"] < 1
            or type(plan["search_completions"]) is not int
            or not 0 <= plan["search_completions"] <= 128):
        raise ValueError("native grammar plan, checkpoint, or authority differs")
    examples = verified_examples(plan, dataset=dataset, seed=seed)
    sources = [hashlib.sha256(example.source_text.encode()).hexdigest() for example in examples]
    from tools.evaluate_semantic_native_grammar import validated_source_pair_map

    expected_pair_map = validated_source_pair_map(
        examples, dataset=dataset, require_contrast=plan.get("source_evidence") == "source_pair_swap")
    if ("source_pair_map" in plan and plan["source_pair_map"] != expected_pair_map
            or plan.get("source_evidence") == "source_pair_swap"
            and "source_pair_map" not in plan):
        raise ValueError("native grammar source-pair map differs")
    source_text_by_sha256 = dict(zip(sources, (example.source_text for example in examples), strict=True))
    forbidden = set(training["fit_ids"]) | set(training["calibration_ids"]) | set(training["held_ids"])
    if (not sources or sources != plan["sources"] or len(set(sources)) != len(sources)
            or forbidden & set(sources) or report["plan_sha256"] != plan["plan_sha256"]
            or report["population"] != len(sources)
            or set(report["row_receipts"]) != set(sources)
            or {path.stem for path in (directory / "rows").glob("*.json")} != set(sources)):
        raise ValueError("native grammar population or source separation differs")
    outcomes, rows = [], []
    tokenizer = None
    if "source_evidence" in plan:
        from mlx_lm.utils import load_tokenizer

        tokenizer = load_tokenizer(Path(training["model_path"]))
    for example, identity in zip(examples, sources, strict=True):
        scored_source = (source_text_by_sha256[expected_pair_map[identity]]
                         if plan.get("source_evidence") == "source_pair_swap"
                         else example.source_text)
        public_values = verified_public_inputs(example)
        row = verified_document(directory / "rows" / f"{identity}.json")
        if ("source_evidence" in plan and row.get("scored_source_sha256")
                != hashlib.sha256(scored_source.encode()).hexdigest()):
            raise ValueError("native grammar scored source differs")
        if plan["schema"].endswith((".v3", ".v4", ".v5", ".v6", ".v7", ".v8", ".v9", ".v10", ".v11", ".v12")):
            from core.learning.semantic_public_inputs import semantic_public_character_inputs

            receipt = semantic_public_character_inputs(example.source_text).receipt()
            if row.get("public_input_receipt_sha256") != receipt["receipt_sha256"]:
                raise ValueError("native grammar public input receipt differs")
        if row["receipt_sha256"] != report["row_receipts"][identity]:
            raise ValueError("native grammar row receipt differs from report")
        verified_prefix_execution(plan, report, row)
        outcomes.append(verify_grammar_row(row, example=example, identity=identity,
                                           plan_sha256=plan["plan_sha256"]))
        if plan["search_mode"] == "greedy":
            if row["search"] is not None:
                raise ValueError("native grammar row search mode differs")
            replay_greedy_decisions(row, example=example, plan=plan, tokenizer=tokenizer,
                                    max_sequence_tokens=training["max_sequence_tokens"],
                                    scored_source=scored_source)
        else:
            if tokenizer is None or plan.get("prefix_strategy") == "trie":
                raise ValueError("native search needs full source-bound scoring")

            def input_receipts_for_choices(choices, *, source=scored_source):
                from core.learning.semantic_native_program import native_text_decision_sequence
                from core.learning.semantic_native_source_control import (
                    apply_native_source_evidence,
                    native_score_input_receipt,
                )

                receipts = []
                for choice in choices:
                    sequence = native_text_decision_sequence(
                        source, choice.text, (choice.span,), tokenizer,
                        max_tokens=training["max_sequence_tokens"])
                    sequence, control = apply_native_source_evidence(
                        sequence, source, tokenizer,
                        mode="source_text" if plan["source_evidence"] == "source_pair_swap"
                        else plan["source_evidence"])
                    receipts.append(native_score_input_receipt(sequence, control))
                return receipts

            def input_receipt_for_program(program, *, source=scored_source):
                from core.learning.semantic_native_codec import native_sequence_for_encoding
                from core.learning.semantic_native_source_control import (
                    apply_native_source_evidence,
                    native_score_input_receipt,
                )

                sequence = native_sequence_for_encoding(source, program, tokenizer,
                    max_tokens=training["max_sequence_tokens"],
                    register_encoding=plan["register_encoding"],
                    decision_basis=training.get("semantic_decision_basis", "program_atoms_v1"))
                sequence, control = apply_native_source_evidence(sequence, source, tokenizer,
                    mode="source_text" if plan["source_evidence"] == "source_pair_swap"
                    else plan["source_evidence"])
                return native_score_input_receipt(sequence, control)

            replay_search_decisions(row, example=example, plan=plan,
                input_types=tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                                  for value in public_values),
                input_receipts_for_choices=input_receipts_for_choices,
                input_receipt_for_program=input_receipt_for_program)
        rows.append(row)
    totals = {"population": len(outcomes),
              "program_equivalent": sum(equivalent for equivalent, _ in outcomes),
              "answer_correct": sum(correct for _, correct in outcomes),
              "bound_forced_completion": sum(row["bound_forced_completion"] for row in rows),
              "depth_bound_reached": sum(row["depth_bound_reached"] for row in rows)}
    if plan["schema"].endswith((".v3", ".v4", ".v5", ".v6", ".v7", ".v8", ".v9", ".v10", ".v11")):
        totals.update(verified_pair_totals(rows, dataset=dataset))
    if any(report[key] != value for key, value in totals.items()):
        raise ValueError("native grammar reported totals differ from execution")
    source_evidence = verified_source_evidence(plan, report, rows)
    drift = sorted(name for name, sha in plan["implementation"].items()
                   if not (ROOT / name).is_file()
                   or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha)
    return {"plan_sha256": plan["plan_sha256"], "report_receipt_sha256": report["receipt_sha256"],
            "weight_mode": weight_mode,
            "source_evidence": source_evidence,
            "input_grounding": input_grounding,
            "dataset": dataset, "seed": seed,
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": scored_checkpoint["receipt_sha256"],
            "totals": totals, "current_implementation_drift": drift,
            "independent_public_value_recovery": True,
            "artifacts_verified": True, "general_transfer_proven": False,
            "broad_gain_proven": False, "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--bundle", action="append")
    parser.add_argument("--meaning-audit", action="store_true",
                        help="also prove denotation or witness differences without changing historical scores")
    args = parser.parse_args()
    if (args.source_report is None) != (args.bundle is None):
        parser.error("source report and every source bundle must be supplied together")
    from tools.evaluate_semantic_native_checkpoint import (
        digest,
        selected_checkpoint,
        verified_document,
    )
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    result = verify_grammar(args.directory, args.training_directory)
    if args.source_report is not None:
        training, _ = selected_checkpoint(args.training_directory)
        plan = verified_document(args.directory / "plan.json", "plan_sha256")
        examples = verified_examples(plan, dataset=result["dataset"], seed=result["seed"])
        if plan["schema"].endswith((".v6", ".v8", ".v9", ".v10", ".v12")):
            from tools.semantic_native_retained_sources import load_retained_native_sources

            _, basis = load_retained_native_sources(args.source_report, args.bundle,
                split=plan["source_cohort_basis"]["split"], count=len(examples))
            if basis != plan["source_cohort_basis"]:
                raise ValueError("retained native verification input basis differs")
            result["source_cohort_basis"] = basis
        else:
            result["source_separation"] = verify_source_separation(
                training, args.source_report, args.bundle, examples)
    version = "v1"
    if args.meaning_audit:
        plan = verified_document(args.directory / "plan.json", "plan_sha256")
        examples = verified_examples(plan, dataset=result["dataset"], seed=result["seed"])
        rows = [verified_document(args.directory / "rows" / f"{identity}.json") for identity in plan["sources"]]
        result["meaning_audit"] = audit_grammar_meanings(rows, examples)
        version = "v2"
    body = {"schema": f"aura.semantic_native_grammar_verification.{version}", **result}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    verified_document(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
