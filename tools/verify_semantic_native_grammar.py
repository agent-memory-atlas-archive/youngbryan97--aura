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
    if (version not in {"v1", "v2", "v3", "v4", "v5", "v6"}
            or plan["schema"] != f"aura.semantic_native_grammar_plan.{version}"
            or report.get("schema") != f"aura.semantic_native_grammar.{version}"):
        raise ValueError("native grammar schema versions differ")
    if version == "v1":
        if "weight_mode" in plan or "weight_mode" in report:
            raise ValueError("historical native grammar weight mode was fitted")
        return "fitted"
    if (plan.get("weight_mode") not in {"fitted", "base"}
            or report.get("weight_mode") != plan["weight_mode"]):
        raise ValueError("native grammar weight mode differs")
    return plan["weight_mode"]


def verified_input_grounding(plan, report):
    version = plan.get("schema", "").rsplit(".", 1)[-1]
    expected = ("semantic_public_character_inputs.v1" if version in {"v3", "v4", "v5", "v6"}
                else "declared_public_inputs")
    if (plan.get("input_grounding") != expected
            or report.get("input_grounding") != plan["input_grounding"]):
        raise ValueError("native grammar input grounding differs")
    return plan["input_grounding"]


def verified_dataset(plan, report):
    version = plan.get("schema", "").rsplit(".", 1)[-1]
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
    if version == "v6":
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
        examples, rebuilt = load_retained_native_sources(basis["source_report_path"],
            [f"{name}={path}" for name, path in sorted(basis["source_manifest_paths"].items())],
            split=basis["split"], count=count)
        if rebuilt != basis:
            raise ValueError("retained native development source reconstruction differs")
        return examples
    if dataset == "natural_request":
        corpus = build_semantic_program_natural_request_corpus(
            seed=seed, examples_per_schema_domain=3,
        )
        ordered = tuple(corpus[index] for sample in range(24)
                        for index in (sample, sample + 24, sample + 48))
    else:
        from tools.semantic_native_operation_interventions import build_native_operation_interventions
        if dataset == "operation_intervention":
            pairs = build_native_operation_interventions(seed=seed)
        elif dataset in {"role_intervention", "dependency_intervention"}:
            from tools.semantic_native_graph_interventions import build_native_graph_interventions

            pairs = build_native_graph_interventions(seed=seed, kind=dataset.removesuffix("_intervention"))
        else:
            from tools.semantic_native_paraphrase_interventions import build_native_paraphrase_interventions

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
    from core.learning.semantic_graph_counterexamples import compare_program_meanings, counterfactual_inputs

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
    input_grounding = verified_input_grounding(plan, report)
    dataset, seed = verified_dataset(plan, report)
    if (plan["schema"].endswith(".v6")
            and plan["source_cohort_basis"]["source_report_sha256"] != training["source_report_sha256"]):
        raise ValueError("retained native source basis differs from the fitted checkpoint")
    if (plan["training_plan_sha256"] != training["plan_sha256"]
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
    examples = verified_examples(plan, dataset=dataset, seed=seed)
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
        verified_public_inputs(example)
        row = verified_document(directory / "rows" / f"{identity}.json")
        if plan["schema"].endswith((".v3", ".v4", ".v5", ".v6")):
            from core.learning.semantic_public_inputs import semantic_public_character_inputs

            receipt = semantic_public_character_inputs(example.source_text).receipt()
            if row.get("public_input_receipt_sha256") != receipt["receipt_sha256"]:
                raise ValueError("native grammar public input receipt differs")
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
    if plan["schema"].endswith((".v3", ".v4", ".v5", ".v6")):
        totals.update(verified_pair_totals(rows, dataset=dataset))
    if any(report[key] != value for key, value in totals.items()):
        raise ValueError("native grammar reported totals differ from execution")
    drift = sorted(name for name, sha in plan["implementation"].items()
                   if not (ROOT / name).is_file()
                   or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha)
    return {"plan_sha256": plan["plan_sha256"], "report_receipt_sha256": report["receipt_sha256"],
            "weight_mode": weight_mode,
            "input_grounding": input_grounding,
            "dataset": dataset, "seed": seed,
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": selected["receipt_sha256"],
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
    from tools.evaluate_semantic_native_checkpoint import digest, selected_checkpoint, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    result = verify_grammar(args.directory, args.training_directory)
    if args.source_report is not None:
        training, _ = selected_checkpoint(args.training_directory)
        plan = verified_document(args.directory / "plan.json", "plan_sha256")
        examples = verified_examples(plan, dataset=result["dataset"], seed=result["seed"])
        if plan["schema"].endswith(".v6"):
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
