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


def replay_greedy_decisions(row, *, example, plan):
    from core.learning.semantic_native_grammar import (
        NativeGrammarIncompleteError,
        decode_native_grammar,
    )

    trace = row["decision_trace"]
    if not isinstance(trace, list):
        raise ValueError("native grammar decision trace is missing")
    consumed = 0

    def recorded_scores(choices):
        nonlocal consumed
        if consumed >= len(trace):
            raise ValueError("native grammar decision trace ended before the graph")
        entry = trace[consumed]
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
                  for value in example.inputs)
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


def verify_grammar(directory, training_directory):
    from core.learning.semantic_program_corpus_natural import (
        build_semantic_program_natural_request_corpus,
    )
    from tools.evaluate_semantic_native_checkpoint import selected_checkpoint, verified_document

    training, selected = selected_checkpoint(training_directory)
    plan = verified_document(directory / "plan.json", "plan_sha256")
    report = verified_document(directory / "report.json")
    if (plan["schema"] != "aura.semantic_native_grammar_plan.v1"
            or report["schema"] != "aura.semantic_native_grammar.v1"
            or plan["training_plan_sha256"] != training["plan_sha256"]
            or plan["checkpoint_receipt_sha256"] != selected["receipt_sha256"]
            or plan["model_descriptor_sha256"] != training["model_descriptor_sha256"]
            or plan["pointer_sha256"] != training["pointer_sha256"]
            or plan["target_available_to_scorer"] is not False
            or report["target_available_to_scorer"] is not False
            or any(plan[key] is not False or report[key] is not False
                   for key in ("serving_authority", "qualification_evidence"))
            or plan["candidate_inventory"] != "none"
            or report["candidate_inventory"] != "none"
            or plan["search_mode"] != "greedy"):
        raise ValueError("native grammar plan, checkpoint, or authority differs")
    examples = build_semantic_program_natural_request_corpus(examples_per_schema_domain=3)
    examples = tuple(examples[index] for sample in range(24)
                     for index in (sample, sample + 24, sample + 48))[:len(plan["sources"])]
    sources = [hashlib.sha256(example.source_text.encode()).hexdigest() for example in examples]
    forbidden = set(training["fit_ids"]) | set(training["calibration_ids"]) | set(training["held_ids"])
    if (not sources or sources != plan["sources"] or len(set(sources)) != len(sources)
            or forbidden & set(sources) or report["plan_sha256"] != plan["plan_sha256"]
            or report["population"] != len(sources)
            or set(report["row_receipts"]) != set(sources)
            or {path.stem for path in (directory / "rows").glob("*.json")} != set(sources)):
        raise ValueError("native grammar population or source separation differs")
    outcomes, rows = [], []
    for example, identity in zip(examples, sources, strict=True):
        row = verified_document(directory / "rows" / f"{identity}.json")
        if row["receipt_sha256"] != report["row_receipts"][identity]:
            raise ValueError("native grammar row receipt differs from report")
        outcomes.append(verify_grammar_row(row, example=example, identity=identity,
                                           plan_sha256=plan["plan_sha256"]))
        if row["search"] is not None:
            raise ValueError("native grammar row search mode differs")
        replay_greedy_decisions(row, example=example, plan=plan)
        rows.append(row)
    totals = {"population": len(outcomes),
              "program_equivalent": sum(equivalent for equivalent, _ in outcomes),
              "answer_correct": sum(correct for _, correct in outcomes),
              "bound_forced_completion": sum(row["bound_forced_completion"] for row in rows),
              "depth_bound_reached": sum(row["depth_bound_reached"] for row in rows)}
    if any(report[key] != value for key, value in totals.items()):
        raise ValueError("native grammar reported totals differ from execution")
    drift = sorted(name for name, sha in plan["implementation"].items()
                   if not (ROOT / name).is_file()
                   or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha)
    return {"plan_sha256": plan["plan_sha256"], "report_receipt_sha256": report["receipt_sha256"],
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": selected["receipt_sha256"],
            "totals": totals, "current_implementation_drift": drift,
            "artifacts_verified": True, "general_transfer_proven": False,
            "broad_gain_proven": False, "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--bundle", action="append")
    args = parser.parse_args()
    if (args.source_report is None) != (args.bundle is None):
        parser.error("source report and every source bundle must be supplied together")
    from tools.evaluate_semantic_native_checkpoint import digest, selected_checkpoint, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    result = verify_grammar(args.directory, args.training_directory)
    if args.source_report is not None:
        from core.learning.semantic_program_corpus_natural import (
            build_semantic_program_natural_request_corpus,
        )
        training, _ = selected_checkpoint(args.training_directory)
        examples = build_semantic_program_natural_request_corpus(examples_per_schema_domain=3)
        examples = tuple(examples[index] for sample in range(24)
                         for index in (sample, sample + 24, sample + 48))[:result["totals"]["population"]]
        result["source_separation"] = verify_source_separation(
            training, args.source_report, args.bundle, examples)
    body = {"schema": "aura.semantic_native_grammar_verification.v1", **result}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    verified_document(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
