#!/usr/bin/env python3
"""Run the frozen native micro stages through the existing detached broker."""

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

from tools.evaluate_semantic_native_checkpoint import digest, verified_document  # noqa: E402


def stage_jobs(*, training_directory, fit_verification, directory, source_report, bundles,
               training_plan_sha256, python, candidate_weight_mode="fitted",
               residual_calibration=None, max_seconds=3600.,
               decision_score_execution="individual", prefix_strategy="full", search_nodes=256):
    """Freeze commands and budgets before any arm can observe an outcome."""
    if ((candidate_weight_mode not in {"fitted", "residual"})
            or (residual_calibration is None) != (candidate_weight_mode == "fitted")):
        raise ValueError("native micro candidate mode lacks its calibration binding")
    if (type(max_seconds) not in {int, float} or not math.isfinite(max_seconds)
            or not 0 < max_seconds <= 14400):
        raise ValueError("native micro decode requires a finite declared runtime bound")
    if decision_score_execution not in {"individual", "causal_groups"}:
        raise ValueError("native micro decision score execution differs")
    if prefix_strategy not in {"full", "trie"} or (prefix_strategy == "trie"
            and decision_score_execution != "causal_groups"):
        raise ValueError("native micro cached prefix needs causal groups")
    if type(search_nodes) is not int or not 1 <= search_nodes <= 256:
        raise ValueError("native micro search node bound differs")
    common = [python, str(ROOT / "tools/evaluate_semantic_native_grammar.py"),
              "--training-directory", str(training_directory), "--max-steps", "8",
              "--search-completions", "4", "--search-nodes", str(search_nodes),
              "--search-score-mode", "native_nonpositive", "--prefix-strategy", prefix_strategy,
              "--max-seconds", str(int(max_seconds) if max_seconds == int(max_seconds) else max_seconds)]
    if decision_score_execution == "causal_groups":
        common += ["--decision-score-execution", decision_score_execution]
    source = ["--source-report", str(source_report)]
    for bundle in bundles:
        source += ["--bundle", bundle]
    stages, plans = [], []
    for stage, cohort, dataset, population in (
            ("reference_requests", "reference", "natural_request", 3),
            ("relation_controls", "controls", "relation_transfer_controls", 9),
            ("retained_requests", "retained", "retained_validation", 6)):
        arms = ("fitted",) if cohort == "controls" else ("fitted", "base", "erasure")
        arm_jobs = []
        for arm in arms:
            output = directory / cohort if cohort == "controls" else directory / cohort / arm
            command = [*common, "--directory", str(output), "--dataset", dataset,
                       "--canary", str(population), "--weight-mode", "base" if arm == "base"
                       else candidate_weight_mode,
                       "--source-evidence", "source_token_erasure" if arm == "erasure" else "source_text"]
            if residual_calibration is not None and arm != "base":
                command += ["--residual-calibration", str(residual_calibration)]
            command += source if cohort == "retained" else [
                "--seed", str(int(training_plan_sha256[:8], 16))]
            name = f"{cohort}-{arm}"
            plan = job(name + "-plan", [*command, "--plan-only"], directory, timeout=300.)
            plans.append(plan)
            verify = [python, str(ROOT / "tools/verify_semantic_native_grammar.py"),
                      "--directory", str(output), "--training-directory", str(training_directory),
                      "--output", str(output / "verification.json"), "--meaning-audit", *source]
            arm_jobs.append({"directory": str(output), "decode": job(name + "-decode", command,
                directory, timeout=max_seconds + 300.), "verify": job(name + "-verify", verify, directory, timeout=1800.)})
        progress_path = directory / (cohort + "-progress.json")
        adjudicate = [python, str(ROOT / "tools/adjudicate_semantic_native_micro_stages.py"),
                     "--training-directory", str(training_directory), "--fit-verification", str(fit_verification),
                     "--reference-root", str(directory / "reference"), "--output", str(progress_path)]
        if cohort != "reference":
            adjudicate += ["--controls-directory", str(directory / "controls")]
        if cohort == "retained":
            adjudicate += ["--retained-root", str(directory / "retained")]
        if residual_calibration is not None:
            adjudicate += ["--residual-calibration", str(residual_calibration)]
        stages.append({"stage": stage, "arms": arm_jobs, "progress_path": str(progress_path),
                       "adjudicate": job(cohort + "-adjudicate", adjudicate, directory, timeout=1800.)})
    return {"plans": plans, "stages": stages}


def job(name, command, directory, *, timeout):
    return {"name": name, "command": command, "cwd": str(ROOT),
            "stdout_path": str(directory / "logs" / (name + ".log")),
            "timeout_s_max": timeout, "max_invocations": 1}


def broker_policy(jobs):
    commands = ([jobs["fit_verify"]] if "fit_verify" in jobs else []) + jobs["plans"]
    for stage in jobs["stages"]:
        commands.extend(action for arm in stage["arms"] for action in (arm["decode"], arm["verify"]))
        commands.append(stage["adjudicate"])
    return [{key: value for key, value in command.items() if key != "name"} for command in commands]


def wait_for_fit(run_directory, *, timeout, inspect=None, clock=time.monotonic, sleep=time.sleep):
    """Advance only after the existing trainer's complete, contained terminal receipt."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("native fit wait requires a positive finite bound")
    if inspect is None:
        from tools.run_detached_step import _status
        inspect = _status
    deadline = clock() + timeout
    while True:
        status = inspect(run_directory)
        if status.get("terminal") is True:
            receipt = status.get("receipt") or {}
            if (receipt.get("passed") is not True or receipt.get("containment_verified") is not True
                    or receipt.get("timed_out") is not False or receipt.get("returncode") != 0
                    or status.get("child_state") != "dead"):
                raise ValueError("native fitting did not complete with proven process cleanup")
            return receipt
        if status.get("supervisor_alive") is not True or status.get("completion_indeterminate") is True:
            raise ValueError("native fitting supervisor cannot prove continued execution")
        remaining = deadline - clock()
        if remaining <= 0:
            raise ValueError("native fit wait reached its declared bound; no model was loaded")
        print(json.dumps({"stage": "fit_wait", "status": status["state"],
                          "heartbeat_sequence": status.get("heartbeat_sequence")}), flush=True)
        sleep(min(30., remaining))


def check_existing_plan(path, *, training, checkpoint, command=None,
                        baseline_checkpoint=None, residual_report_receipt_sha256=None):
    plan = verified_document(path, "plan_sha256")
    mode = None if command is None else command[command.index("--weight-mode") + 1]
    expected_checkpoint = baseline_checkpoint if mode == "base" and baseline_checkpoint else checkpoint
    if (plan["training_plan_sha256"] != training["plan_sha256"]
            or plan["checkpoint_receipt_sha256"] != expected_checkpoint["receipt_sha256"]
            or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
                   for name, sha in plan["implementation"].items())):
        raise ValueError("native micro continuation changed its frozen candidate or code")
    if command is not None:
        if "--max-seconds" in command and plan.get("max_seconds") != float(
                command[command.index("--max-seconds") + 1]):
            raise ValueError("native micro continuation changed its frozen runtime bound")
        if "--search-nodes" in command and plan.get("search_nodes") != int(
                command[command.index("--search-nodes") + 1]):
            raise ValueError("native micro continuation changed its frozen search bound")
        execution = (command[command.index("--decision-score-execution") + 1]
                     if "--decision-score-execution" in command else "individual")
        if plan.get("decision_score_execution", "individual") != execution:
            raise ValueError("native micro continuation changed its decision score execution")
        prefix_strategy = (command[command.index("--prefix-strategy") + 1]
                           if "--prefix-strategy" in command else "full")
        if plan.get("prefix_strategy", "full") != prefix_strategy:
            raise ValueError("native micro continuation changed its prefix strategy")
        contract = plan.get("residual_calibration")
        if plan.get("weight_mode") != mode:
            raise ValueError("native micro continuation changed its candidate mode")
        if mode == "residual":
            expected_directory = Path(command[command.index("--residual-calibration") + 1]).resolve()
            if (not isinstance(contract, dict)
                    or contract.get("directory") != str(expected_directory)
                    or contract.get("report_receipt_sha256") != residual_report_receipt_sha256):
                raise ValueError("native micro continuation changed its source calibration")
        elif contract is not None:
            raise ValueError("native micro non-residual arm acquired a calibration")


def require_learned_checkpoint(checkpoint):
    if type(checkpoint.get("step")) is not int or checkpoint["step"] <= 0:
        raise ValueError("native micro decode requires a selected learned checkpoint")


def run_stages(jobs, *, training, checkpoint, invoke, baseline_checkpoint=None,
               residual_report_receipt_sha256=None):
    """Reuse completed decodes; a failed acceptance never starts the next stage."""
    for plan_job in jobs["plans"]:
        command = plan_job["command"]
        path = Path(command[command.index("--directory") + 1]) / "plan.json"
        if not path.exists():
            invoke(plan_job)
        check_existing_plan(path, training=training, checkpoint=checkpoint, command=command,
            baseline_checkpoint=baseline_checkpoint,
            residual_report_receipt_sha256=residual_report_receipt_sha256)
    for stage in jobs["stages"]:
        for arm in stage["arms"]:
            output = Path(arm["directory"])
            if not (output / "report.json").exists():
                if list((output / "rows").glob("*.json")):
                    raise ValueError("partial native decode requires a new declared attempt; no silent retry")
                invoke(arm["decode"])
            verified_document(output / "report.json")
            invoke(arm["verify"])
            verification = verified_document(output / "verification.json")
            if (verification.get("artifacts_verified") is not True
                    or verification.get("current_implementation_drift") != []):
                raise ValueError("native micro arm lacks current independent verification")
        invoke(stage["adjudicate"])
        progress = verified_document(Path(stage["progress_path"]))
        measured = next(item for item in progress["stages"] if item["stage"] == stage["stage"])
        print(json.dumps({"stage": stage["stage"], "status": measured["status"],
                          "progress_receipt_sha256": progress["receipt_sha256"]}), flush=True)
        if measured["status"] != "passed":
            return progress
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-directory", "fit-verification", "directory", "source-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--policy-output", type=Path,
                        help="freeze exact broker commands without running or loading a model")
    parser.add_argument("--wait-fit-supervisor", type=Path)
    parser.add_argument("--fit-bank", type=Path)
    parser.add_argument("--fit-parent", type=Path)
    parser.add_argument("--fit-wait-seconds", type=float, default=10800.)
    parser.add_argument("--candidate-weight-mode", choices=("fitted", "residual"), default="fitted")
    parser.add_argument("--residual-calibration", type=Path)
    parser.add_argument("--max-seconds", type=float, default=3600.,
                        help="per-arm model-active bound, frozen into the pipeline and broker policy")
    parser.add_argument("--decision-score-execution", choices=("individual", "causal_groups"),
                        default="individual")
    parser.add_argument("--prefix-strategy", choices=("full", "trie"), default="full")
    args = parser.parse_args()
    if ((args.wait_fit_supervisor is None) != (args.fit_bank is None)
            or (args.wait_fit_supervisor is None) != (args.fit_parent is None)
            or not math.isfinite(args.fit_wait_seconds) or args.fit_wait_seconds <= 0):
        parser.error("supervised fit handoff requires its bank, parent, and finite wait bound")
    if (args.residual_calibration is None) != (args.candidate_weight_mode == "fitted"):
        parser.error("residual micro decode requires a calibration directory")
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.directory / "pipeline.json")
    (args.directory / "logs").mkdir(parents=True, exist_ok=True)
    training = verified_document(args.training_directory / "plan.json", "plan_sha256")
    jobs = stage_jobs(training_directory=args.training_directory, fit_verification=args.fit_verification,
        directory=args.directory, source_report=args.source_report, bundles=args.bundle,
        training_plan_sha256=training["plan_sha256"], python=sys.executable,
        candidate_weight_mode=args.candidate_weight_mode,
        residual_calibration=args.residual_calibration, max_seconds=args.max_seconds,
        decision_score_execution=args.decision_score_execution,
        prefix_strategy=args.prefix_strategy)
    if args.wait_fit_supervisor is not None:
        command = [sys.executable, str(ROOT / "tools/verify_semantic_native_fit.py"),
                   "--directory", str(args.training_directory), "--bank", str(args.fit_bank),
                   "--parent", str(args.fit_parent), "--source-report", str(args.source_report),
                   "--output", str(args.fit_verification)]
        for bundle in args.bundle:
            command += ["--bundle", bundle]
        jobs["fit_verify"] = job("fit-verify", command, args.directory, timeout=3600.)
    body = {"schema": "aura.native_micro_pipeline.v2" if args.residual_calibration else
            "aura.native_micro_pipeline.v1", "training_plan_sha256": training["plan_sha256"],
            "fit_verification": str(args.fit_verification), "jobs": jobs,
            **({"residual_calibration": str(args.residual_calibration)}
               if args.residual_calibration else {}),
            "fit_supervisor": None if args.wait_fit_supervisor is None else str(args.wait_fit_supervisor),
            "fit_wait_seconds": args.fit_wait_seconds,
            "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
            "serving_authority": False, "general_transfer_proven": False}
    _save_if_absent(args.directory / "pipeline.json", {**body, "receipt_sha256": digest(body)})
    if args.policy_output is not None:
        _save_if_absent(args.policy_output, broker_policy(jobs))
        print(json.dumps({"stage": "policy_only", "commands": len(broker_policy(jobs)),
                          "training_plan_sha256": training["plan_sha256"]}), flush=True)
        return 0
    from core.runtime.detached_subprocess_broker import broker_available, run_brokered_process
    from tools.adjudicate_semantic_native_micro_stages import verified_native_candidate

    if not broker_available():
        raise ValueError("native micro stages require the existing detached supervisor broker")
    def invoke(item):
        print(json.dumps({"stage": item["name"], "status": "started"}), flush=True)
        outcome = run_brokered_process(item["command"], cwd=ROOT,
            stdout_path=Path(item["stdout_path"]), timeout_s=item["timeout_s_max"])
        if (outcome.returncode != 0 or outcome.status != "passed" or outcome.timed_out
                or not outcome.containment_verified):
            raise ValueError(f"native micro command failed: {item['name']}:{outcome.status}")
        print(json.dumps({"stage": item["name"], "status": "process_completed",
                          "broker_receipt_sha256": outcome.receipt_sha256}), flush=True)

    if args.wait_fit_supervisor is not None:
        receipt = wait_for_fit(args.wait_fit_supervisor, timeout=args.fit_wait_seconds)
        print(json.dumps({"stage": "fit_wait", "status": "completed",
                          "supervisor_receipt_sha256": receipt["receipt_sha256"]}), flush=True)
        invoke(jobs["fit_verify"])
    training, checkpoint, _, residual = verified_native_candidate(
        args.training_directory, args.fit_verification, args.residual_calibration)
    require_learned_checkpoint(checkpoint)
    from tools.evaluate_semantic_native_checkpoint import selected_checkpoint
    _, baseline_checkpoint = selected_checkpoint(args.training_directory)
    result = run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke,
        baseline_checkpoint=baseline_checkpoint,
        residual_report_receipt_sha256=None if residual is None else residual["report_receipt_sha256"])
    return 0 if result["full_development_ready"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
