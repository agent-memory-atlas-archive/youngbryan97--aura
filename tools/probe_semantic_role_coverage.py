#!/usr/bin/env python3
"""Reject an irrelevant role candidate before model-active native decoding.

This development diagnostic reads frozen source plans and target annotations
only after the rule bank has been fitted. It does not score a model, select a
checkpoint, or grant transfer authority.
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

SCHEMA = "aura.semantic_role_coverage.v1"


def cohort_coverage(bank, examples) -> dict:
    """Count first-step bindings without passing annotations to the bank."""
    rows = []
    for example in examples:
        if not example.instructions:
            raise ValueError("role coverage needs an annotated first instruction")
        first = example.instructions[0].instruction
        expected = tuple(first.args)
        proposed = bank.first_input_binding(example.source_text, first.op)
        outcome = "abstain" if proposed is None else "correct" if proposed == expected else "wrong"
        rows.append({"source_sha256": hashlib.sha256(example.source_text.encode()).hexdigest(),
                     "construction": example.construction_id,
                     "operation": first.op, "expected": expected,
                     "proposed": proposed, "outcome": outcome})
    return {"population": len(rows),
            "correct": sum(row["outcome"] == "correct" for row in rows),
            "wrong": sum(row["outcome"] == "wrong" for row in rows),
            "abstain": sum(row["outcome"] == "abstain" for row in rows),
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--role-rule-bank", type=Path, required=True)
    parser.add_argument("--evaluation-plan", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("role coverage output already exists")
    from core.learning.semantic_role_rule_induction import RoleRuleBank
    from core.runtime.file_write_gateway import get_file_write_gateway
    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.fit_semantic_role_rule_bank import read_role_rule_fit
    from tools.verify_semantic_native_grammar import verified_examples

    training = verified_document(args.training_directory / "plan.json", "plan_sha256")
    plans = [verified_document(path, "plan_sha256") for path in args.evaluation_plan]
    if len({plan["plan_sha256"] for plan in plans}) != len(plans):
        raise ValueError("role coverage plans must be distinct")
    sources = tuple(source for plan in plans for source in plan["sources"])
    if len(set(sources)) != len(sources):
        raise ValueError("role coverage source populations overlap")
    fit = read_role_rule_fit(args.role_rule_bank, training=training,
                             held_source_sha256s=sources)
    bank = RoleRuleBank.from_dict(fit["bank"])
    cohorts = []
    for path, plan in zip(args.evaluation_plan, plans, strict=True):
        if plan["training_plan_sha256"] != training["plan_sha256"]:
            raise ValueError("role coverage plan has another training ancestry")
        examples = verified_examples(plan, dataset=plan["dataset"], seed=plan["seed"])
        actual = tuple(hashlib.sha256(example.source_text.encode()).hexdigest()
                       for example in examples)
        if actual != tuple(plan["sources"]):
            raise ValueError("role coverage source plan does not reconstruct")
        cohorts.append({"plan_path": str(path.resolve()),
                        "plan_sha256": plan["plan_sha256"],
                        "dataset": plan["dataset"], **cohort_coverage(bank, examples)})
    body = {"schema": SCHEMA, "fit_receipt_sha256": fit["receipt_sha256"],
            "training_plan_sha256": training["plan_sha256"],
            "cohorts": cohorts, "model_results_measured": False,
            "general_transfer_proven": False, "serving_authority": False}
    payload = {**body, "receipt_sha256": digest(body)}
    get_file_write_gateway().write_json(args.output, payload, schema_version=1,
        schema_name=SCHEMA, source="probe_semantic_role_coverage")
    print(json.dumps({"receipt_sha256": payload["receipt_sha256"],
                      "cohorts": [{key: cohort[key] for key in
                                   ("dataset", "population", "correct", "wrong", "abstain")}
                                  for cohort in cohorts]}))


if __name__ == "__main__":
    main()
