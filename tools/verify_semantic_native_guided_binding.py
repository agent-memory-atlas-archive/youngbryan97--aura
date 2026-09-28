#!/usr/bin/env python3
"""Replay conditional base decisions without granting end-to-end gain evidence."""

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


def replay_guided_row(row, guide_row, *, example, guide_plan, tokenizer, max_tokens):
    from core.learning.semantic_native_grammar import decode_native_grammar
    from core.learning.semantic_native_program import native_text_decision_sequence
    from core.learning.semantic_native_source_control import native_score_input_receipt
    from tools.verify_semantic_native_grammar import (
        verified_prefix_execution,
        verified_public_inputs,
        verified_trie_anchor,
    )

    trace, guide_trace = row["decisions"], guide_row["decision_trace"]
    if (not isinstance(trace, list) or len(trace) != len(guide_trace)
            or len(row["score_input_receipts"]) != len(trace)
            or row["guide_row_receipt_sha256"] != guide_row["receipt_sha256"]):
        raise ValueError("guided native trace population or guide receipt differs")
    verified_prefix_execution(guide_plan, {"prefix_strategy": "trie"}, row)
    consumed, anchor = 0, None

    def scores(choices):
        nonlocal consumed, anchor
        if consumed >= len(trace):
            raise ValueError("guided native trace exhausted")
        entry, guide = trace[consumed], guide_trace[consumed]
        values = [choice.value for choice in choices]
        measured = entry["base_scores"]
        if (entry["choices"] != values or guide["choices"] != values
                or entry["kind"] != guide["kind"]
                or entry["guide_chosen"] != guide["chosen"]
                or guide["chosen"] not in values
                or not isinstance(measured, list) or len(measured) != len(values)
                or any(type(score) not in (int, float) or not math.isfinite(score)
                       for score in measured)
                or entry["base_winner"] != values[max(range(len(measured)), key=measured.__getitem__)]):
            raise ValueError("guided native choices, scores, or winner differs")
        sequences = tuple(native_text_decision_sequence(
            example.source_text, choice.text, (choice.span,), tokenizer,
            max_tokens=max_tokens) for choice in choices)
        expected = [native_score_input_receipt(sequence, None) for sequence in sequences]
        if row["score_input_receipts"][consumed] != expected:
            raise ValueError("guided native source tokens differ")
        anchor = verified_trie_anchor(sequences, row["prefix_execution"], anchor)
        consumed += 1
        return tuple(float(value == guide["chosen"]) for value in values)

    types = tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                  for value in verified_public_inputs(example))
    decoded = decode_native_grammar(types, scores, max_steps=guide_plan["max_steps"],
                                    register_encoding=guide_plan["register_encoding"])
    if (consumed != len(trace) or anchor is None
            or decoded.program.to_dict() != guide_row["program"]
            or [entry["kind"] for entry in decoded.trace] != [entry["kind"] for entry in trace]):
        raise ValueError("guided native replay differs from guide program")
    return {kind: {"matched": sum(entry["base_winner"] == entry["guide_chosen"]
                                  for entry in trace if entry["kind"] == kind),
                   "total": sum(entry["kind"] == kind for entry in trace)}
            for kind in ("operation", "reference", "termination")}


def verify_guided(directory, training_directory, guide_directory):
    from mlx_lm.utils import load_tokenizer

    from tools.evaluate_semantic_native_checkpoint import verified_document
    from tools.probe_semantic_native_guided_binding import guide_basis
    from tools.verify_semantic_native_grammar import verified_examples

    training, selected, guide_plan, guide_report, guide_rows = guide_basis(
        training_directory, guide_directory)
    plan = verified_document(directory / "plan.json", "plan_sha256")
    report = verified_document(directory / "report.json")
    pinned = {"training_plan_sha256": training["plan_sha256"],
              "checkpoint_receipt_sha256": selected["receipt_sha256"],
              "guide_plan_sha256": guide_plan["plan_sha256"],
              "guide_report_receipt_sha256": guide_report["receipt_sha256"],
              "guide_row_receipts": guide_report["row_receipts"],
              "model_descriptor_sha256": training["model_descriptor_sha256"],
              "pointer_sha256": training["pointer_sha256"],
              "dataset": guide_plan["dataset"], "seed": guide_plan["seed"],
              "sources": guide_plan["sources"],
              "guidance": "fitted_target_blind_trace_all_prior_choices",
              "cohort_selection": "all_exact_fitted_guide_rows",
              "measured_weight_mode": "base", "precision": "float32",
              "prefix_strategy": "trie", "target_available_to_scorer": False,
              "serving_authority": False, "qualification_evidence": False}
    if (plan["schema"] != "aura.native_guided_binding_plan.v1"
            or any(plan.get(key) != value for key, value in pinned.items())
            or report["schema"] != "aura.native_guided_binding.v1"
            or report["plan_sha256"] != plan["plan_sha256"]
            or report.get("guidance_is_diagnostic_only") is not True
            or any(report.get(key) is not False for key in
                   ("serving_authority", "qualification_evidence"))):
        raise ValueError("guided native plan, guide, or authority differs")
    sources = plan["sources"]
    if (report["population"] != len(sources) or set(report["row_receipts"]) != set(sources)
            or {path.stem for path in (directory / "rows").glob("*.json")} != set(sources)):
        raise ValueError("guided native population differs")
    examples = verified_examples(guide_plan, dataset=guide_plan["dataset"], seed=guide_plan["seed"])
    tokenizer = load_tokenizer(Path(training["model_path"]))
    totals = {kind: {"matched": 0, "total": 0} for kind in ("operation", "reference", "termination")}
    for identity, example, guide_row in zip(sources, examples, guide_rows, strict=True):
        row = verified_document(directory / "rows" / f"{identity}.json")
        if (row["schema"] != "aura.native_guided_binding_row.v1"
                or row["source_sha256"] != identity or row["plan_sha256"] != plan["plan_sha256"]
                or row["receipt_sha256"] != report["row_receipts"][identity]):
            raise ValueError("guided native row identity differs")
        measured = replay_guided_row(row, guide_row, example=example, guide_plan=guide_plan,
                                    tokenizer=tokenizer, max_tokens=training["max_sequence_tokens"])
        for kind, counts in measured.items():
            for key, value in counts.items():
                totals[kind][key] += value
    if report["base_matches_fitted_by_kind"] != totals:
        raise ValueError("guided native reported totals differ")
    drift = sorted(name for name, sha in plan["implementation"].items()
                   if not (ROOT / name).is_file()
                   or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha)
    return {"plan_sha256": plan["plan_sha256"], "report_receipt_sha256": report["receipt_sha256"],
            "population": len(sources), "base_matches_fitted_by_kind": totals,
            "current_implementation_drift": drift, "artifacts_verified": True,
            "model_scores_recomputed": False, "guidance_is_diagnostic_only": True,
            "end_to_end_gain_proven": False, "general_transfer_proven": False,
            "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("directory", "training-directory", "guide-directory", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    body = verify_guided(args.directory, args.training_directory, args.guide_directory)
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    print(json.dumps(body), flush=True)


if __name__ == "__main__":
    main()
