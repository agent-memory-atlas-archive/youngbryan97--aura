#!/usr/bin/env python3
"""Run a small, matched reference screen before the full native micro gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.evaluate_semantic_native_checkpoint import digest, verified_document  # noqa: E402
from tools.run_semantic_native_micro_stages import (  # noqa: E402
    check_existing_plan,
    stage_jobs,
)

SCHEMA = "aura.native_reference_screen.v1"


def reference_screen_jobs(jobs):
    """Keep only the three matched reference arms from the shared job builder."""
    plans = jobs["plans"][:3]
    arms = jobs["stages"][0]["arms"]
    if ([item["name"] for item in plans] != [
            "reference-fitted-plan", "reference-base-plan", "reference-erasure-plan"]
            or [Path(item["directory"]).name for item in arms]
            != ["fitted", "base", "erasure"]):
        raise ValueError("native screen reference job inventory changed")
    return plans, arms


def screen_policy(plans, arms):
    commands = [*plans, *(item for arm in arms for item in (arm["decode"], arm["verify"]))]
    return [{key: value for key, value in item.items() if key != "name"}
            for item in commands]


def screen_verdict(comparison):
    from tools.adjudicate_semantic_native_micro_stages import cohort_passed

    return (cohort_passed(comparison, population=3)
            and comparison["source_intervention"]["source_dependent_gains"] > 0)


def check_screen_plan_contract(path, command, training):
    from core.learning.semantic_native_codec import register_encoding_from_plan
    from tools.evaluate_semantic_native_grammar import grammar_examples

    plan = verified_document(path, "plan_sha256")
    def flag(name):
        return command[command.index(name) + 1]

    seed = int(flag("--seed"))
    sources = [hashlib.sha256(example.source_text.encode()).hexdigest()
               for example in grammar_examples(dataset="natural_request", seed=seed, count=3)]
    expected = {"dataset": "natural_request", "seed": seed, "sources": sources,
                "source_pair_map": {},
                "source_evidence": flag("--source-evidence"),
                "max_steps": int(flag("--max-steps")),
                "search_nodes": int(flag("--search-nodes")),
                "search_completions": int(flag("--search-completions")),
                "search_score_mode": flag("--search-score-mode"),
                "search_mode": "best_first_then_complete_graph_score",
                "register_encoding": register_encoding_from_plan(training),
                "candidate_inventory": "none",
                "input_grounding": "semantic_public_character_inputs.v1",
                "model_descriptor_sha256": training["model_descriptor_sha256"],
                "pointer_sha256": training["pointer_sha256"],
                "target_available_to_scorer": False,
                "held_labels_used_for_fit_or_selection": False}
    if (any(plan.get(key) != value for key, value in expected.items())
            or plan.get("prefix_strategy", "full") != flag("--prefix-strategy")):
        raise ValueError("native screen saved plan changed its source or search contract")


def run_screen(plans, arms, *, training, checkpoint, baseline_checkpoint,
               residual_report_receipt_sha256, invoke, compare):
    for item in plans:
        command = item["command"]
        path = Path(command[command.index("--directory") + 1]) / "plan.json"
        if not path.exists():
            invoke(item)
        check_existing_plan(path, training=training, checkpoint=checkpoint,
            baseline_checkpoint=baseline_checkpoint, command=command,
            residual_report_receipt_sha256=residual_report_receipt_sha256)
        check_screen_plan_contract(path, command, training)
    for arm in arms:
        output = Path(arm["directory"])
        if not (output / "report.json").exists():
            if list((output / "rows").glob("*.json")):
                raise ValueError("partial native screen decode requires a new declared attempt")
            invoke(arm["decode"])
        verified_document(output / "report.json")
        verification_path = output / "verification.json"
        if not verification_path.exists():
            invoke(arm["verify"])
        verification = verified_document(verification_path)
        if (verification.get("artifacts_verified") is not True
                or verification.get("current_implementation_drift") != []):
            raise ValueError("native screen arm lacks independent verification")
    result = compare()
    return {"screen_passed": screen_verdict(result), "comparison": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-directory", "fit-verification", "directory", "source-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--candidate-weight-mode", choices=("fitted", "residual"), default="fitted")
    parser.add_argument("--residual-calibration", type=Path)
    parser.add_argument("--decision-score-execution", choices=("individual", "causal_groups"),
                        default="causal_groups")
    parser.add_argument("--search-nodes", type=int, default=16)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--policy-output", type=Path)
    args = parser.parse_args()
    if (args.residual_calibration is None) != (args.candidate_weight_mode == "fitted"):
        parser.error("residual candidate requires its source-only calibration")
    from tools.adjudicate_semantic_native_micro_stages import verified_native_candidate
    from tools.compare_semantic_native_grammar_fit import compare_directories
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.directory / "screen.json")
    (args.directory / "logs").mkdir(parents=True, exist_ok=True)
    training, checkpoint, fit, residual = verified_native_candidate(
        args.training_directory, args.fit_verification, args.residual_calibration)
    source_report_sha256 = hashlib.sha256(args.source_report.read_bytes()).hexdigest()
    if source_report_sha256 != training["source_report_sha256"]:
        raise ValueError("native screen source report differs from verified training")
    jobs = stage_jobs(training_directory=args.training_directory,
        fit_verification=args.fit_verification, directory=args.directory,
        source_report=args.source_report, bundles=args.bundle,
        training_plan_sha256=training["plan_sha256"], python=sys.executable,
        candidate_weight_mode=args.candidate_weight_mode,
        residual_calibration=args.residual_calibration,
        max_seconds=args.max_seconds,
        decision_score_execution=args.decision_score_execution,
        search_nodes=args.search_nodes)
    plans, arms = reference_screen_jobs(jobs)
    policy = screen_policy(plans, arms)
    implementation_paths = ("tools/run_semantic_native_reference_screen.py",
                            "tools/run_semantic_native_micro_stages.py",
                            "tools/adjudicate_semantic_native_micro_stages.py",
                            "tools/compare_semantic_native_grammar_fit.py")
    body = {"schema": SCHEMA, "training_plan_sha256": training["plan_sha256"],
            "candidate_checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
            "fit_verification_receipt_sha256": fit["receipt_sha256"],
            "residual_report_receipt_sha256": None if residual is None else residual["report_receipt_sha256"],
            "source_report_sha256": source_report_sha256,
            "implementation": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in implementation_paths},
            "search_nodes": args.search_nodes, "max_seconds": args.max_seconds,
            "commands": policy, "screen_only": True, "serving_authority": False}
    _save_if_absent(args.directory / "screen-plan.json",
                    {**body, "plan_sha256": digest(body)})
    if args.policy_output is not None:
        _save_if_absent(args.policy_output, policy)
        print(json.dumps({"stage": "policy_only", "commands": len(policy),
                          "plan_sha256": digest(body)}), flush=True)
        return 0
    from core.runtime.detached_subprocess_broker import broker_available, run_brokered_process
    from tools.evaluate_semantic_native_checkpoint import selected_checkpoint

    if not broker_available():
        raise ValueError("native reference screen requires the detached supervisor broker")
    _, baseline_checkpoint = selected_checkpoint(args.training_directory)

    def invoke(item):
        print(json.dumps({"stage": item["name"], "status": "started"}), flush=True)
        outcome = run_brokered_process(item["command"], cwd=ROOT,
            stdout_path=Path(item["stdout_path"]), timeout_s=item["timeout_s_max"])
        if (outcome.returncode != 0 or outcome.status != "passed" or outcome.timed_out
                or not outcome.containment_verified):
            raise ValueError(f"native screen command failed: {item['name']}:{outcome.status}")

    result = run_screen(plans, arms, training=training, checkpoint=checkpoint,
        baseline_checkpoint=baseline_checkpoint,
        residual_report_receipt_sha256=None if residual is None else residual["report_receipt_sha256"],
        invoke=invoke, compare=lambda: compare_directories(
            fitted_directory=args.directory / "reference/fitted",
            base_directory=args.directory / "reference/base",
            erasure_directory=args.directory / "reference/erasure",
            training_directory=args.training_directory))
    outcome = {"schema": "aura.native_reference_screen_result.v1",
               "plan_sha256": digest(body), "comparison_receipt_sha256": result["comparison"]["receipt_sha256"],
               "screen_passed": result["screen_passed"],
               "full_budget_required": True, "general_transfer_proven": False,
               "serving_authority": False}
    _save_if_absent(args.directory / "screen.json", {**outcome, "receipt_sha256": digest(outcome)})
    print(json.dumps(outcome), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
