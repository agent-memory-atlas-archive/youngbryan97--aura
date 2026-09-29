#!/usr/bin/env python3
"""Advance the complete development stage through fixed, verified windows."""

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

from tools.evaluate_semantic_native_checkpoint import digest, verified_document  # noqa: E402
from tools.run_semantic_native_micro_stages import check_existing_plan, job, wait_for_fit  # noqa: E402


def development_jobs(*, training_directory, fit_verification, directory, micro_root,
                     micro_checkout, source_report, bundles, python, window_size, max_seconds):
    if (type(window_size) is not int or not 1 <= window_size <= 500
            or type(max_seconds) not in {int, float}
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400):
        raise ValueError("native development windows require fixed finite population and time bounds")
    source = ["--source-report", str(source_report)]
    for bundle in bundles:
        source += ["--bundle", bundle]
    common = [python, str(ROOT / "tools/evaluate_semantic_native_grammar.py"),
              "--training-directory", str(training_directory), "--max-steps", "8",
              "--search-completions", "4", "--search-nodes", "256",
              "--search-score-mode", "native_nonpositive", "--prefix-strategy", "full",
              "--max-seconds", str(max_seconds), "--dataset", "retained_validation", *source]
    plans, windows = [], []
    for offset in range(0, 500, window_size):
        count = min(window_size, 500 - offset)
        root = directory / "windows" / f"{offset:04d}-{offset + count:04d}"
        arms = []
        for arm in ("fitted", "base", "erasure"):
            output = root / arm
            command = [*common, "--directory", str(output), "--source-offset", str(offset),
                       "--canary", str(count), "--weight-mode", "base" if arm == "base" else "fitted",
                       "--source-evidence", "source_token_erasure" if arm == "erasure" else "source_text"]
            name = f"window-{offset:04d}-{arm}"
            plans.append(job(name + "-plan", [*command, "--plan-only"], directory, timeout=300.))
            verify = [python, str(ROOT / "tools/verify_semantic_native_grammar.py"),
                      "--directory", str(output), "--training-directory", str(training_directory),
                      "--output", str(output / "verification.json"), "--meaning-audit", *source]
            arms.append({"directory": str(output),
                "decode": job(name + "-decode", command, directory, timeout=max_seconds + 300.),
                "verify": job(name + "-verify", verify, directory, timeout=1800.)})
        compare = [python, str(ROOT / "tools/compare_semantic_native_grammar_fit.py"),
                   "--training-directory", str(training_directory), "--fitted-directory", str(root / "fitted"),
                   "--base-directory", str(root / "base"), "--erasure-directory", str(root / "erasure"),
                   "--output", str(root / "comparison.json")]
        windows.append({"offset": offset, "count": count, "directory": str(root), "arms": arms,
                        "compare": job(f"window-{offset:04d}-compare", compare, directory, timeout=1800.)})
    micro = [python, str(micro_checkout / "tools/adjudicate_semantic_native_micro_stages.py"),
             "--training-directory", str(training_directory), "--fit-verification", str(fit_verification),
             "--reference-root", str(micro_root / "reference"), "--controls-directory", str(micro_root / "controls"),
             "--retained-root", str(micro_root / "retained"), "--output", str(directory / "micro-progress.json")]
    micro_job = job("micro-cpu-replay", micro, directory, timeout=3600.)
    micro_job["cwd"] = str(micro_checkout)
    adjudicate = [python, str(ROOT / "tools/adjudicate_semantic_native_development.py"),
                  "--training-directory", str(training_directory), "--fit-verification", str(fit_verification),
                  "--output", str(directory / "development-progress.json")]
    for window in windows:
        adjudicate += ["--window-root", window["directory"]]
    return {"micro_replay": micro_job, "plans": plans, "windows": windows,
            "adjudicate": job("development-adjudicate", adjudicate, directory, timeout=14400.)}


def development_policy(jobs):
    commands = [jobs["micro_replay"], *jobs["plans"]]
    for window in jobs["windows"]:
        commands.extend(action for arm in window["arms"] for action in (arm["decode"], arm["verify"]))
        commands.append(window["compare"])
    commands.append(jobs["adjudicate"])
    return [{key: value for key, value in item.items() if key != "name"} for item in commands]


def run_windows(jobs, *, training, checkpoint, invoke):
    from tools.adjudicate_semantic_native_micro_stages import cohort_passed
    from tools.probe_semantic_proposer_crossfit import _save_if_absent

    for item in jobs["plans"]:
        command = item["command"]
        path = Path(command[command.index("--directory") + 1]) / "plan.json"
        if not path.exists():
            invoke(item)
        check_existing_plan(path, training=training, checkpoint=checkpoint)
    for window in jobs["windows"]:
        for arm in window["arms"]:
            output = Path(arm["directory"])
            if not (output / "report.json").exists():
                if list((output / "rows").glob("*.json")):
                    raise ValueError("partial native window requires a new declared attempt")
                invoke(arm["decode"])
            verified_document(output / "report.json")
            invoke(arm["verify"])
            verified = verified_document(output / "verification.json")
            if (verified.get("artifacts_verified") is not True
                    or verified.get("current_implementation_drift") != []):
                raise ValueError("native development window lacks independent current verification")
        invoke(window["compare"])
        comparison = verified_document(Path(window["directory"]) / "comparison.json")
        passed = cohort_passed(comparison, population=window["count"])
        body = {"schema": "aura.native_development_window_progress.v1",
                "offset": window["offset"], "population": window["count"],
                "comparison_receipt_sha256": comparison["receipt_sha256"],
                "training_plan_sha256": training["plan_sha256"],
                "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
                "status": "passed" if passed else "failed", "current_stage": "full_development",
                "serving_authority": False}
        _save_if_absent(Path(window["directory"]) / "acceptance.json",
                        {**body, "receipt_sha256": digest(body)})
        print(json.dumps(body), flush=True)
        if not passed:
            return False
    invoke(jobs["adjudicate"])
    command = jobs["adjudicate"]["command"]
    output = Path(command[command.index("--output") + 1])
    return verified_document(output)["full_development_passed"] is True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("training-directory", "fit-verification", "directory", "micro-root",
                 "micro-checkout", "micro-supervisor", "source-report"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--window-size", type=int, required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("--micro-wait-seconds", type=float, default=86400.)
    parser.add_argument("--policy-output", type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.micro_wait_seconds) or args.micro_wait_seconds <= 0:
        parser.error("native micro wait requires a positive finite bound")
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.directory / "pipeline.json")
    (args.directory / "logs").mkdir(parents=True, exist_ok=True)
    training = verified_document(args.training_directory / "plan.json", "plan_sha256")
    jobs = development_jobs(training_directory=args.training_directory, fit_verification=args.fit_verification,
        directory=args.directory, micro_root=args.micro_root, micro_checkout=args.micro_checkout,
        source_report=args.source_report, bundles=args.bundle, python=sys.executable,
        window_size=args.window_size, max_seconds=args.max_seconds)
    body = {"schema": "aura.native_development_pipeline.v1", "jobs": jobs,
            "training_plan_sha256": training["plan_sha256"],
            "micro_supervisor": str(args.micro_supervisor), "micro_wait_seconds": args.micro_wait_seconds,
            "source_report_sha256": hashlib.sha256(args.source_report.read_bytes()).hexdigest(),
            "serving_authority": False}
    _save_if_absent(args.directory / "pipeline.json", {**body, "receipt_sha256": digest(body)})
    if args.policy_output is not None:
        _save_if_absent(args.policy_output, development_policy(jobs))
        print(json.dumps({"stage": "policy_only", "windows": len(jobs["windows"]),
                          "population": 500, "commands": len(development_policy(jobs))}), flush=True)
        return 0
    from core.runtime.detached_subprocess_broker import broker_available, run_brokered_process
    from tools.adjudicate_semantic_native_micro_stages import verified_native_fit

    if not broker_available():
        raise ValueError("native development requires the existing detached supervisor broker")

    def invoke(item):
        print(json.dumps({"stage": item["name"], "status": "started"}), flush=True)
        result = run_brokered_process(item["command"], cwd=Path(item["cwd"]),
            stdout_path=Path(item["stdout_path"]), timeout_s=item["timeout_s_max"])
        if (result.returncode != 0 or result.status != "passed" or result.timed_out
                or not result.containment_verified):
            raise ValueError(f"native development command failed: {item['name']}:{result.status}")

    wait_for_fit(args.micro_supervisor, timeout=args.micro_wait_seconds)
    invoke(jobs["micro_replay"])
    expected = verified_document(args.micro_root / "retained-progress.json")
    replay = verified_document(args.directory / "micro-progress.json")
    if expected != replay or replay.get("full_development_ready") is not True:
        raise ValueError("native development requires the unchanged passed micro stage")
    training, checkpoint, _ = verified_native_fit(args.training_directory, args.fit_verification)
    if (replay["training_plan_sha256"] != training["plan_sha256"]
            or replay["checkpoint_receipt_sha256"] != checkpoint["receipt_sha256"]):
        raise ValueError("native development changed the micro-qualified candidate")
    return 0 if run_windows(jobs, training=training, checkpoint=checkpoint, invoke=invoke) else 2


if __name__ == "__main__":
    raise SystemExit(main())
