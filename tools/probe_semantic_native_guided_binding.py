#!/usr/bin/env python3
"""Score unfitted grammar choices on the same target-blind fitted prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def guided_choice_scores(choices, guide_entry, measured_scores):
    """Preserve measured scores while forcing only the verified guide path."""
    values = [choice.value for choice in choices]
    if (values != guide_entry["choices"] or len(values) != len(measured_scores)
            or any(type(score) not in (int, float) or not math.isfinite(score)
                   for score in measured_scores)
            or guide_entry["chosen"] not in values):
        raise ValueError("guided native choices differ from the fitted trace")
    chosen = values.index(guide_entry["chosen"])
    return tuple(float(index == chosen) for index in range(len(values)))


def guide_basis(training_directory, guide_directory, *, require_current=True):
    from tools.evaluate_semantic_native_checkpoint import selected_checkpoint, verified_document
    from tools.verify_semantic_native_grammar import verify_grammar

    training, selected = selected_checkpoint(training_directory)
    verified = verify_grammar(guide_directory, training_directory)
    plan = verified_document(guide_directory / "plan.json", "plan_sha256")
    report = verified_document(guide_directory / "report.json")
    if (require_current and verified["current_implementation_drift"]
            or plan["schema"] != "aura.semantic_native_grammar_plan.v7"
            or plan["weight_mode"] != "fitted" or plan["source_evidence"] != "source_text"
            or plan["search_mode"] != "greedy" or plan["prefix_strategy"] != "trie"
            or verified["totals"]["program_equivalent"] != report["population"]
            or plan["training_plan_sha256"] != training["plan_sha256"]):
        raise ValueError("guided native scorer needs one current exact fitted arm")
    rows = tuple(verified_document(guide_directory / "rows" / f"{identity}.json")
                 for identity in plan["sources"])
    return training, selected, plan, report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--guide-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 3600:
        parser.error("guided native evaluation needs a finite runtime")

    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.directory / "report.json")
    training, selected, guide_plan, guide_report, guide_rows = guide_basis(
        args.training_directory, args.guide_directory)
    from tools.semantic_native_execution import execution_from_plan

    execution = execution_from_plan(training, check_installed=True)
    if execution is None or execution["precision"] != "float32":
        raise ValueError("guided native scorer needs measured float32 arithmetic")
    from core.brain.llm.model_registry import get_active_cortex_spec

    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("guided native scorer model differs from training")
    paths = ("tools/probe_semantic_native_guided_binding.py",
             "tools/verify_semantic_native_guided_binding.py",
             "tools/semantic_native_execution.py",
             "tools/evaluate_semantic_native_grammar.py",
             "tools/verify_semantic_native_grammar.py",
             "core/learning/semantic_native_grammar.py",
             "core/learning/semantic_native_program.py",
             "core/learning/frozen_prefix_branches.py",
             "tools/train_semantic_native_program.py")
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in paths}
    body = {"schema": "aura.native_guided_binding_plan.v1",
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": selected["receipt_sha256"],
            "guide_plan_sha256": guide_plan["plan_sha256"],
            "guide_report_receipt_sha256": guide_report["receipt_sha256"],
            "guide_row_receipts": guide_report["row_receipts"],
            "model_descriptor_sha256": spec.descriptor_sha256,
            "pointer_sha256": spec.pointer_sha256,
            "dataset": guide_plan["dataset"], "seed": guide_plan["seed"],
            "sources": guide_plan["sources"], "max_seconds": args.max_seconds,
            "implementation": implementation,
            "guidance": "fitted_target_blind_trace_all_prior_choices",
            "cohort_selection": "all_exact_fitted_guide_rows",
            "measured_weight_mode": "base", "precision": "float32",
            "prefix_strategy": "trie", "target_available_to_scorer": False,
            "serving_authority": False, "qualification_evidence": False}
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "population": len(guide_rows),
                          "plan_sha256": plan["plan_sha256"]}), flush=True)
        return

    from mlx_lm import load

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.semantic_native_grammar import decode_native_grammar
    from core.learning.semantic_native_program import native_text_decision_sequence
    from core.learning.semantic_native_source_control import native_score_input_receipt
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.evaluate_semantic_native_grammar import (
        capture_grammar_choices,
        grammar_examples,
        source_input_types,
    )
    from tools.semantic_native_execution import apply_execution
    from tools.train_semantic_native_program import native_loss

    examples = grammar_examples(dataset=guide_plan["dataset"], seed=guide_plan["seed"],
                                count=len(guide_rows))
    started, results = time.monotonic(), []
    with (standalone_model_lane(owner_id=f"native-guided-binding:{args.directory.name}",
            model_path=str(spec.model_path), purpose="evaluation", preemptible=False,
            require_exclusive=True, allow_owner_eviction=False,
            metadata={"production_effect": False}), mlx_memory_envelope(fraction=.8)):
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        apply_execution(model, training)
        split = len(model.layers) - training["suffix_layers"]
        prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
        for identity, example, guide_row in zip(plan["sources"], examples, guide_rows, strict=True):
            if hashlib.sha256(example.source_text.encode()).hexdigest() != identity:
                raise ValueError("guided native source population differs")
            _values, types = source_input_types(example.source_text)
            decisions, input_receipts = [], []
            branches = None

            def score(choices, *, example=example, guide_row=guide_row,
                      decisions=decisions, input_receipts=input_receipts):
                nonlocal branches
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("guided native scorer reached its finite runtime bound")
                ordinal = len(decisions)
                if ordinal >= len(guide_row["decision_trace"]):
                    raise ValueError("guided native trace ended before the graph")
                guide_entry = guide_row["decision_trace"][ordinal]
                sequences = tuple(native_text_decision_sequence(
                    example.source_text, choice.text, (choice.span,), tokenizer,
                    max_tokens=training["max_sequence_tokens"]) for choice in choices)
                states, branches = capture_grammar_choices(prefix, branches, model, sequences,
                    split_at=split, max_tokens=training["max_sequence_tokens"], strategy="trie")
                scores = tuple(-native_loss(suffix, hidden, sequence, summed=True,
                    scope="semantic_decisions").item()
                    for hidden, sequence in zip(states, sequences, strict=True))
                forced = guided_choice_scores(choices, guide_entry, scores)
                values = [choice.value for choice in choices]
                decisions.append({"kind": guide_entry["kind"], "choices": values,
                                  "guide_chosen": guide_entry["chosen"],
                                  "base_scores": scores,
                                  "base_winner": values[max(range(len(scores)), key=scores.__getitem__)]})
                input_receipts.append([native_score_input_receipt(sequence, None)
                                       for sequence in sequences])
                return forced

            generated = decode_native_grammar(types, score,
                max_steps=guide_plan["max_steps"], register_encoding=guide_plan["register_encoding"])
            if (generated.program.to_dict() != guide_row["program"]
                    or len(decisions) != len(guide_row["decision_trace"])
                    or [entry["kind"] for entry in generated.trace]
                    != [entry["kind"] for entry in decisions]):
                raise ValueError("guided native graph differs from the verified fitted trace")
            row_body = {"schema": "aura.native_guided_binding_row.v1",
                        "plan_sha256": plan["plan_sha256"], "source_sha256": identity,
                        "guide_row_receipt_sha256": guide_row["receipt_sha256"],
                        "decisions": decisions, "score_input_receipts": input_receipts,
                        "prefix_execution": branches.receipt()}
            row = {**row_body, "receipt_sha256": digest(row_body)}
            _save_if_absent(args.directory / "rows" / f"{identity}.json", row)
            results.append(row)
            print(json.dumps({"stage": "guided_row", "observed": len(results),
                              "population": len(guide_rows)}), flush=True)
    kinds = {kind: (sum(entry["kind"] == kind and entry["base_winner"] == entry["guide_chosen"]
                        for row in results for entry in row["decisions"]),
                    sum(entry["kind"] == kind for row in results for entry in row["decisions"]))
             for kind in ("operation", "reference", "termination")}
    result = {"schema": "aura.native_guided_binding.v1", "plan_sha256": plan["plan_sha256"],
              "population": len(results),
              "row_receipts": {row["source_sha256"]: row["receipt_sha256"] for row in results},
              "base_matches_fitted_by_kind": {kind: {"matched": value[0], "total": value[1]}
                                               for kind, value in kinds.items()},
              "elapsed_seconds": time.monotonic() - started,
              "guidance_is_diagnostic_only": True, "serving_authority": False,
              "qualification_evidence": False}
    _save_if_absent(args.directory / "report.json", {**result, "receipt_sha256": digest(result)})
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
