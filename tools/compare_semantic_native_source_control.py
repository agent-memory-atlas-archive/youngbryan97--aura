#!/usr/bin/env python3
"""Adjudicate a matched target-blind source-content lesion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def matched_source_control(full_plan: dict, erased_plan: dict,
                           full_report: dict, erased_report: dict,
                           full_rows: tuple[dict, ...], erased_rows: tuple[dict, ...]) -> dict:
    """Compare outcome changes without treating model influence as correctness."""
    if (full_plan.get("source_evidence") != "source_text"
            or erased_plan.get("source_evidence") != "source_token_erasure"):
        raise ValueError("source-control comparison needs intact and erased source arms")
    excluded = {"source_evidence", "plan_sha256"}
    if ({key: value for key, value in full_plan.items() if key not in excluded}
            != {key: value for key, value in erased_plan.items() if key not in excluded}):
        raise ValueError("source-control arms differ beyond source visibility")
    sources = full_plan.get("sources")
    if (not isinstance(sources, list) or not sources or sources != erased_plan.get("sources")
            or full_report.get("plan_sha256") != full_plan.get("plan_sha256")
            or erased_report.get("plan_sha256") != erased_plan.get("plan_sha256")
            or full_report.get("population") != len(sources)
            or erased_report.get("population") != len(sources)
            or len(full_rows) != len(sources) or len(erased_rows) != len(sources)):
        raise ValueError("source-control arms do not cover the same population")
    for arm, rows in (("source_text", full_rows), ("source_token_erasure", erased_rows)):
        if ([row.get("source_sha256") for row in rows] != sources
                or any(row.get("source_evidence") != arm for row in rows)):
            raise ValueError("source-control row coverage or intervention differs")
    return {
        "population": len(sources),
        "full_correct": sum(row["program_equivalent"] is True for row in full_rows),
        "erased_correct": sum(row["program_equivalent"] is True for row in erased_rows),
        "full_answer_correct": sum(row["answer_correct"] is True for row in full_rows),
        "erased_answer_correct": sum(row["answer_correct"] is True for row in erased_rows),
        "full_only": sum(full["program_equivalent"] is True
                         and erased["program_equivalent"] is False
                         for full, erased in zip(full_rows, erased_rows, strict=True)),
        "erased_only": sum(full["program_equivalent"] is False
                           and erased["program_equivalent"] is True
                           for full, erased in zip(full_rows, erased_rows, strict=True)),
        "program_changed": sum(full["program"] != erased["program"]
                               for full, erased in zip(full_rows, erased_rows, strict=True)),
        "full_pair_exact": full_report.get("pair_exact"),
        "erased_pair_exact": erased_report.get("pair_exact"),
        "full_source_responsive": full_report.get("source_responsive"),
        "erased_source_responsive": erased_report.get("source_responsive"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-directory", required=True, type=Path)
    parser.add_argument("--erased-directory", required=True, type=Path)
    parser.add_argument("--training-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    from tools.evaluate_semantic_native_checkpoint import digest, verified_document
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.verify_semantic_native_grammar import verify_grammar

    configure_refit_environment(args.output)
    verified = [verify_grammar(directory, args.training_directory)
                for directory in (args.full_directory, args.erased_directory)]
    plans = [verified_document(directory / "plan.json", "plan_sha256")
             for directory in (args.full_directory, args.erased_directory)]
    reports = [verified_document(directory / "report.json")
               for directory in (args.full_directory, args.erased_directory)]
    rows = [tuple(verified_document(directory / "rows" / f"{identity}.json")
                  for identity in plan["sources"])
            for directory, plan in zip((args.full_directory, args.erased_directory), plans, strict=True)]
    comparison = matched_source_control(plans[0], plans[1], reports[0], reports[1], rows[0], rows[1])
    body = {"schema": "aura.semantic_native_source_control_comparison.v1",
            "full_report_receipt_sha256": verified[0]["report_receipt_sha256"],
            "erased_report_receipt_sha256": verified[1]["report_receipt_sha256"],
            "training_plan_sha256": verified[0]["training_plan_sha256"],
            "checkpoint_receipt_sha256": verified[0]["checkpoint_receipt_sha256"],
            "comparison": comparison,
            "source_effect_observed": comparison["program_changed"] > 0,
            "causal_correctness_proven": False,
            "general_transfer_proven": False,
            "serving_authority": False}
    result = {**body, "receipt_sha256": digest(body)}
    _save_if_absent(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
