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


def _matched_source_arm(full_plan: dict, altered_plan: dict,
                        full_report: dict, altered_report: dict,
                        full_rows: tuple[dict, ...], altered_rows: tuple[dict, ...],
                        *, altered_mode: str) -> dict:
    if (full_plan.get("source_evidence") != "source_text"
            or altered_plan.get("source_evidence") != altered_mode):
        raise ValueError("source-control comparison needs intact and intervened source arms")
    excluded = {"source_evidence", "plan_sha256"}
    if ({key: value for key, value in full_plan.items() if key not in excluded}
            != {key: value for key, value in altered_plan.items() if key not in excluded}):
        raise ValueError("source-control arms differ beyond source visibility")
    sources = full_plan.get("sources")
    if (not isinstance(sources, list) or not sources or sources != altered_plan.get("sources")
            or full_report.get("plan_sha256") != full_plan.get("plan_sha256")
            or altered_report.get("plan_sha256") != altered_plan.get("plan_sha256")
            or full_report.get("population") != len(sources)
            or altered_report.get("population") != len(sources)
            or len(full_rows) != len(sources) or len(altered_rows) != len(sources)):
        raise ValueError("source-control arms do not cover the same population")
    pair_map = full_plan.get("source_pair_map", {})
    if altered_mode == "source_pair_swap" and (
            not isinstance(pair_map, dict) or set(pair_map) != set(sources)
            or set(pair_map.values()) != set(sources)):
        raise ValueError("source-control swap partner coverage differs")
    for arm, rows in (("source_text", full_rows), (altered_mode, altered_rows)):
        if ([row.get("source_sha256") for row in rows] != sources
                or any(row.get("source_evidence") != arm for row in rows)
                or ("source_pair_map" in full_plan and any(
                    row.get("scored_source_sha256") != (
                        pair_map[row["source_sha256"]] if arm == "source_pair_swap"
                        else row["source_sha256"])
                    for row in rows))):
            raise ValueError("source-control row coverage or intervention differs")
    return {
        "population": len(sources),
        "full_correct": sum(row["program_equivalent"] is True for row in full_rows),
        "altered_original_correct": sum(row["program_equivalent"] is True for row in altered_rows),
        "full_answer_correct": sum(row["answer_correct"] is True for row in full_rows),
        "altered_original_answer_correct": sum(row["answer_correct"] is True
                                               for row in altered_rows),
        "full_only": sum(full["program_equivalent"] is True
                         and altered["program_equivalent"] is False
                         for full, altered in zip(full_rows, altered_rows, strict=True)),
        "altered_only": sum(full["program_equivalent"] is False
                            and altered["program_equivalent"] is True
                            for full, altered in zip(full_rows, altered_rows, strict=True)),
        "program_changed": sum(full["program"] != altered["program"]
                               for full, altered in zip(full_rows, altered_rows, strict=True)),
        "full_pair_exact": full_report.get("pair_exact"),
        "altered_original_pair_exact": altered_report.get("pair_exact"),
        "full_source_responsive": full_report.get("source_responsive"),
        "altered_original_source_responsive": altered_report.get("source_responsive"),
    }


def matched_source_control(full_plan: dict, erased_plan: dict,
                           full_report: dict, erased_report: dict,
                           full_rows: tuple[dict, ...], erased_rows: tuple[dict, ...]) -> dict:
    """Compare erasure outcomes without treating influence as correctness."""
    generic = _matched_source_arm(full_plan, erased_plan, full_report, erased_report,
                                  full_rows, erased_rows,
                                  altered_mode="source_token_erasure")
    renamed = {"altered_original_correct": "erased_correct",
               "altered_original_answer_correct": "erased_answer_correct",
               "altered_only": "erased_only",
               "altered_original_pair_exact": "erased_pair_exact",
               "altered_original_source_responsive": "erased_source_responsive"}
    return {renamed.get(key, key): value for key, value in generic.items()}


def matched_source_swap(full_plan: dict, swap_plan: dict,
                        full_report: dict, swap_report: dict,
                        full_rows: tuple[dict, ...], swap_rows: tuple[dict, ...],
                        examples: tuple) -> dict:
    """Grade the natural-source swap against the partner's target after decode."""
    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent

    comparison = _matched_source_arm(full_plan, swap_plan, full_report, swap_report,
                                     full_rows, swap_rows, altered_mode="source_pair_swap")
    sources = full_plan["sources"]
    pair_map = full_plan.get("source_pair_map")
    if (not isinstance(pair_map, dict) or not pair_map
            or len(examples) != len(sources)
            or len(sources) % 2
            or any(pair_map.get(sources[index]) != sources[index ^ 1]
                   for index in range(len(sources)))
            or any(tuple(examples[index].inputs) != tuple(examples[index ^ 1].inputs)
                   for index in range(len(sources)))):
        raise ValueError("source-control swap partner coverage differs")
    by_source = dict(zip(sources, examples, strict=True))
    partner_correct = []
    partner_answer_correct = []
    for row in swap_rows:
        partner = by_source[pair_map[row["source_sha256"]]]
        payload = row["program"]
        decoded = (Program(len(partner.inputs), tuple(
            Instruction(op, tuple(arguments)) for op, arguments in payload["instructions"]))
            if row["decode_status"] == "completed" and payload is not None else None)
        partner_correct.append(decoded is not None and
                               semantic_programs_structurally_equivalent(decoded, partner.program))
        try:
            partner_answer_correct.append(decoded is not None and
                                          decoded.run(partner.inputs) == partner.program.run(partner.inputs))
        except (ValueError, TypeError, RuntimeError, ArithmeticError, IndexError):
            partner_answer_correct.append(False)
    comparison.update({
        "swapped_partner_correct": sum(partner_correct),
        "swapped_partner_answer_correct": sum(partner_answer_correct),
        "exact_swap_pairs": sum(
            all(full_rows[index + offset]["program_equivalent"] is True
                and partner_correct[index + offset]
                for offset in (0, 1))
            for index in range(0, len(sources), 2)),
        "swapped_matches_partner_full_program": sum(
            swap_rows[index]["program"] is not None
            and full_rows[index ^ 1]["program"] is not None
            and swap_rows[index]["program"] == full_rows[index ^ 1]["program"]
            for index in range(len(sources))),
    })
    return comparison


def matched_verification_identity(full: dict, altered: dict, *, altered_mode="source_token_erasure") -> None:
    """Reject a paired claim when either arm was verified under changed code."""
    if (full.get("current_implementation_drift") != []
            or altered.get("current_implementation_drift") != []
            or full.get("training_plan_sha256") != altered.get("training_plan_sha256")
            or full.get("checkpoint_receipt_sha256") != altered.get("checkpoint_receipt_sha256")
            or full.get("source_evidence") != "source_text"
            or altered.get("source_evidence") != altered_mode):
        raise ValueError("source-control verification identity or implementation differs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-directory", required=True, type=Path)
    parser.add_argument("--erased-directory", required=True, type=Path)
    parser.add_argument("--swapped-directory", type=Path)
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
    matched_verification_identity(*verified)
    plans = [verified_document(directory / "plan.json", "plan_sha256")
             for directory in (args.full_directory, args.erased_directory)]
    reports = [verified_document(directory / "report.json")
               for directory in (args.full_directory, args.erased_directory)]
    rows = [tuple(verified_document(directory / "rows" / f"{identity}.json")
                  for identity in plan["sources"])
            for directory, plan in zip((args.full_directory, args.erased_directory), plans, strict=True)]
    comparison = matched_source_control(plans[0], plans[1], reports[0], reports[1], rows[0], rows[1])
    swapped = None
    if args.swapped_directory is not None:
        from tools.verify_semantic_native_grammar import verified_dataset, verified_examples

        swapped_verification = verify_grammar(args.swapped_directory, args.training_directory)
        matched_verification_identity(verified[0], swapped_verification,
                                      altered_mode="source_pair_swap")
        swap_plan = verified_document(args.swapped_directory / "plan.json", "plan_sha256")
        swap_report = verified_document(args.swapped_directory / "report.json")
        swap_rows = tuple(verified_document(args.swapped_directory / "rows" / f"{identity}.json")
                          for identity in swap_plan["sources"])
        dataset, seed = verified_dataset(plans[0], reports[0])
        examples = tuple(verified_examples(plans[0], dataset=dataset, seed=seed))
        swapped = matched_source_swap(plans[0], swap_plan, reports[0], swap_report,
                                      rows[0], swap_rows, examples)
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
    if swapped is not None:
        body["schema"] = "aura.semantic_native_source_control_comparison.v2"
        body["swapped_report_receipt_sha256"] = swapped_verification["report_receipt_sha256"]
        body["swap_comparison"] = swapped
    result = {**body, "receipt_sha256": digest(body)}
    _save_if_absent(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
