#!/usr/bin/env python3
"""Regrade a target-blind native decode from its durable source and programs."""

from __future__ import annotations

import argparse
import hashlib
import json
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
            or report["candidate_inventory"] != "none"):
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
        if (plan["search_mode"] == "greedy" and row["search"] is not None) or (
                plan["search_mode"] != "greedy" and row["search"] is None):
            raise ValueError("native grammar row search mode differs")
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
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    result = verify_grammar(args.directory, args.training_directory)
    body = {"schema": "aura.semantic_native_grammar_verification.v1", **result}
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    verified_document(args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
