#!/usr/bin/env python3
"""Recompute smaller native search budgets from verified, saved model scores.

This is a retrospective development screen. It makes no new model call and
does not substitute for a prospectively frozen matched-budget evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "aura.semantic_native_budget_frontier.v1"


def replay_budget(row: dict, example, plan: dict, *, max_nodes: int) -> dict:
    """Replay only previously visited choices and rank their saved graphs."""
    from core.learning.semantic_native_search import search_native_grammar
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent
    from tools.evaluate_semantic_native_grammar import source_input_types

    full = row["search"]
    transcript = full["score_transcript"]
    if not 1 <= max_nodes <= plan["search_nodes"]:
        raise ValueError("budget must be inside the measured search")
    consumed = 0

    def saved_scores(choices):
        nonlocal consumed
        if consumed >= len(transcript):
            raise ValueError("budget replay needs an unmeasured decision")
        entry = transcript[consumed]
        if entry["choices"] != [choice.value for choice in choices]:
            raise ValueError("budget replay changed the typed choice inventory")
        consumed += 1
        return tuple(entry["scores"])

    types = source_input_types(example.source_text)[1]
    result = search_native_grammar(types, saved_scores,
        max_steps=plan["max_steps"], max_nodes=max_nodes,
        completions=plan["search_completions"],
        register_encoding=plan["register_encoding"],
        score_mode=plan["search_score_mode"])
    proposals = [candidate.result.program.to_dict() for candidate in result.candidates]
    if proposals != [item["program"] for item in full["proposals"][:len(proposals)]]:
        raise ValueError("budget replay changed the saved proposal prefix")
    graph_scores = full["complete_graph_scores"][:len(proposals)]
    if len(graph_scores) != len(proposals):
        raise ValueError("budget replay lacks whole-graph scores")
    selected = (None if not graph_scores else result.candidates[
        max(range(len(graph_scores)), key=graph_scores.__getitem__)].result.program)
    exact = selected is not None and semantic_programs_structurally_equivalent(
        selected, example.program)
    answer = False
    if selected is not None:
        try:
            answer = selected.run(example.inputs) == example.program.run(example.inputs)
        except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
            pass
    if max_nodes == plan["search_nodes"] and (
            consumed != len(transcript) or selected is None and row["program"] is not None
            or selected is not None and selected.to_dict() != row["program"]
            or exact is not row["program_equivalent"]
            or answer is not row["answer_correct"]):
        raise ValueError("full-budget replay differs from measured decode")
    return {"max_nodes": max_nodes, "expanded_nodes": result.expanded_nodes,
            "scored_decisions": consumed, "candidates": len(proposals),
            "scored_alternatives": result.scored_alternatives,
            "requested_top_k_proven": result.requested_top_k_proven,
            "procedure_equivalent": exact, "answer_correct": answer}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--budget", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("budget-frontier output already exists")

    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.verify_semantic_native_grammar import verified_examples

    plan = verified_document(args.directory / "plan.json", "plan_sha256")
    report = verified_document(args.directory / "report.json")
    verification = verified_document(args.directory / "verification.json")
    budgets = sorted(set(args.budget) | {plan["search_nodes"]})
    if (plan.get("search_mode") != "best_first_then_complete_graph_score"
            or any(not 1 <= budget <= plan["search_nodes"] for budget in budgets)
            or report.get("plan_sha256") != plan["plan_sha256"]
            or verification.get("artifacts_verified") is not True
            or verification.get("current_implementation_drift") != []
            or verification.get("plan_sha256") != plan["plan_sha256"]
            or verification.get("report_receipt_sha256") != report["receipt_sha256"]):
        raise ValueError("budget frontier needs a complete source-matched verification")
    examples = verified_examples(plan, dataset=plan["dataset"], seed=plan["seed"])
    identities = [hashlib.sha256(example.source_text.encode()).hexdigest()
                  for example in examples]
    if (identities != plan["sources"] or set(report["row_receipts"]) != set(identities)):
        raise ValueError("budget frontier source inventory differs")
    rows = []
    for identity, example in zip(identities, examples, strict=True):
        row = verified_document(args.directory / "rows" / f"{identity}.json")
        if (row["source_sha256"] != identity
                or row["receipt_sha256"] != report["row_receipts"][identity]
                or row["plan_sha256"] != plan["plan_sha256"]):
            raise ValueError("budget frontier row receipt differs")
        rows.append({"source_sha256": identity,
                     "budgets": [replay_budget(row, example, plan, max_nodes=budget)
                                 for budget in budgets]})
    totals = [{"max_nodes": budget,
               "population": len(rows),
               "exact_procedures": sum(item["budgets"][index]["procedure_equivalent"]
                                       for item in rows),
               "correct_answers": sum(item["budgets"][index]["answer_correct"]
                                      for item in rows),
               "scored_decisions": sum(item["budgets"][index]["scored_decisions"]
                                       for item in rows),
               "scored_alternatives": sum(item["budgets"][index]["scored_alternatives"]
                                         for item in rows)}
              for index, budget in enumerate(budgets)]
    implementation_paths = ("tools/replay_semantic_native_budget_frontier.py",
                            "core/learning/semantic_native_search.py",
                            "core/learning/semantic_native_grammar.py")
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                      for name in implementation_paths}
    body = {"schema": SCHEMA, "plan_sha256": plan["plan_sha256"],
            "report_receipt_sha256": report["receipt_sha256"],
            "verification_receipt_sha256": verification["receipt_sha256"],
            "implementation": implementation,
            "historical_implementation_drift": sorted(name for name in implementation_paths
                if name in plan["implementation"] and
                implementation[name] != plan["implementation"][name]),
            "budgets": budgets, "totals": totals, "rows": rows,
            "model_scores_reused": True, "prospective_decode_measured": False,
            "general_transfer_proven": False, "serving_authority": False}
    payload = {**body, "receipt_sha256": digest(body)}
    get_file_write_gateway().write_json(args.output, payload, schema_version=1,
        schema_name=SCHEMA, source="replay_semantic_native_budget_frontier")
    print(json.dumps({"receipt_sha256": payload["receipt_sha256"], "totals": totals}))


if __name__ == "__main__":
    main()
