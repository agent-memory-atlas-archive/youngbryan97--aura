#!/usr/bin/env python3
"""Project a verified native grammar trace onto a shorter depth bound."""

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


def project_decisions(trace, input_types, *, max_steps, register_encoding):
    """Replay only score vectors the original run actually measured."""
    from core.learning.semantic_native_grammar import (
        NativeGrammarIncompleteError,
        decode_native_grammar,
    )

    if (not isinstance(trace, list) or not trace or type(max_steps) is not int
            or max_steps < 1):
        raise ValueError("native depth projection needs a saved trace and positive bound")
    consumed = 0

    def recorded_scores(choices):
        nonlocal consumed
        if consumed == len(trace):
            raise ValueError("native depth projection exhausted its original decisions")
        entry = trace[consumed]
        consumed += 1
        values = [choice.value for choice in choices]
        scores = entry["scores"]
        if (entry["choices"] != values or not isinstance(scores, list)
                or len(scores) != len(values)
                or any(type(score) not in (int, float) or not math.isfinite(score)
                       for score in scores)
                or entry["chosen"] != values[max(range(len(scores)), key=scores.__getitem__)]):
            raise ValueError("native depth projection changed a scored choice")
        return tuple(scores)

    try:
        result = decode_native_grammar(input_types, recorded_scores,
            max_steps=max_steps, register_encoding=register_encoding)
    except NativeGrammarIncompleteError as failure:
        program, projected_trace = failure.program, failure.trace
        status, forced = "disconnected_at_depth_bound", False
    else:
        program, projected_trace = result.program, result.trace
        status, forced = "completed", result.bound_forced_completion
    if json.loads(json.dumps(projected_trace)) != trace[:consumed]:
        raise ValueError("native depth projection changed an original score prefix")
    return program, status, forced, consumed


def project_verified_run(directory, training_directory, *, max_steps):
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent
    from tools.evaluate_semantic_native_checkpoint import verified_document
    from tools.verify_semantic_native_grammar import (
        verified_dataset,
        verified_examples,
        verified_pair_totals,
        verified_public_inputs,
        verify_grammar,
    )

    original = verify_grammar(directory, training_directory)
    plan = verified_document(directory / "plan.json", "plan_sha256")
    report = verified_document(directory / "report.json")
    if (original["current_implementation_drift"] or type(max_steps) is not int
            or not 1 <= max_steps < plan["max_steps"]):
        raise ValueError("native depth projection requires intact evidence and a shorter bound")
    dataset, seed = verified_dataset(plan, report)
    examples = verified_examples(plan, dataset=dataset, seed=seed)
    rows = []
    for example in examples:
        identity = hashlib.sha256(example.source_text.encode()).hexdigest()
        source_row = verified_document(directory / "rows" / f"{identity}.json")
        public_inputs = verified_public_inputs(example)
        input_types = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                            for value in public_inputs)
        program, status, forced, consumed = project_decisions(
            source_row["decision_trace"], input_types, max_steps=max_steps,
            register_encoding=plan["register_encoding"])
        exact = (status == "completed"
                 and semantic_programs_structurally_equivalent(program, example.program))
        answer = False
        if status == "completed":
            try:
                answer = program.run(public_inputs) == example.program.run(example.inputs)
            except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
                pass
        rows.append({"source_sha256": identity,
                     "original_row_receipt_sha256": source_row["receipt_sha256"],
                     "consumed_original_decisions": consumed,
                     "original_decisions": len(source_row["decision_trace"]),
                     "program": program.to_dict(), "decode_status": status,
                     "bound_forced_completion": forced,
                     "depth_bound_reached": forced or status == "disconnected_at_depth_bound",
                     "program_equivalent": exact, "answer_correct": answer})
    totals = {"population": len(rows),
              "program_equivalent": sum(row["program_equivalent"] for row in rows),
              "answer_correct": sum(row["answer_correct"] for row in rows),
              "bound_forced_completion": sum(row["bound_forced_completion"] for row in rows),
              "depth_bound_reached": sum(row["depth_bound_reached"] for row in rows)}
    totals.update(verified_pair_totals(rows, dataset=dataset))
    return {"schema": "aura.semantic_native_depth_projection.v1",
            "original_plan_sha256": plan["plan_sha256"],
            "original_report_receipt_sha256": report["receipt_sha256"],
            "weight_mode": original["weight_mode"], "dataset": dataset,
            "original_max_steps": plan["max_steps"], "projected_max_steps": max_steps,
            "score_source": "verified_original_model_trace",
            "new_model_decode_performed": False,
            "target_available_to_scorer": False,
            "serving_authority": False, "qualification_evidence": False,
            "totals": totals, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    result = project_verified_run(args.directory, args.training_directory,
                                  max_steps=args.max_steps)
    _save_if_absent(args.output, {**result, "receipt_sha256": digest(result)})
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     sort_keys=True))


if __name__ == "__main__":
    main()
