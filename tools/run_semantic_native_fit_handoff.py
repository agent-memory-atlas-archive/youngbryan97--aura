#!/usr/bin/env python3
"""Run the prepared native protocol only after its supervised prerequisites finish."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def verified_preparation_document(path: Path) -> dict:
    from core.learning.semantic_fit_checkpoint import fit_identity

    value = json.loads(path.read_bytes())
    if value.get("receipt_sha256") != fit_identity({key: item for key, item in value.items()
                                                  if key != "receipt_sha256"}):
        raise ValueError("native preparation handoff digest differs")
    return value


def preparation_paths(handoff: dict) -> dict:
    """Remove only the two preparation modes from the same bound training command."""
    from core.learning.semantic_fit_checkpoint import fit_identity

    jobs = handoff.get("jobs")
    if (handoff.get("schema") != "aura.semantic_source_native_preparation_plan.v1"
            or handoff.get("receipt_sha256") != fit_identity({key: value for key, value in handoff.items()
                                                         if key != "receipt_sha256"})
            or not isinstance(jobs, list) or len(jobs) != 2
            or any(not isinstance(row, dict) for row in jobs)
            or [row.get("name") for row in jobs] != ["native-plan", "native-supervision"]):
        raise ValueError("native fit handoff requires the bound preparation protocol")
    left, right = (row.get("command") for row in jobs)
    if (not isinstance(left, list) or not isinstance(right, list) or len(left) < 4
            or any(not isinstance(value, str) for value in left + right)
            or left[:-1] != right[:-1] or left[-1] != "--plan-only"
            or right[-1] != "--supervision-only"
            or Path(left[1]).name != "train_semantic_native_program.py"
            or any(flag in left[:-1] for flag in ("--plan-only", "--supervision-only"))
            or "--require-identifiable-supervision" not in left):
        raise ValueError("native fit handoff changed the prepared training command")
    if jobs[0].get("cwd") != jobs[1].get("cwd"):
        raise ValueError("native preparation commands disagree on their source checkout")
    cwd = Path(jobs[0]["cwd"])
    if not cwd.is_absolute():
        raise ValueError("native preparation source checkout must be absolute")

    def path(flag):
        indices = [index for index, value in enumerate(left[:-1]) if value == flag]
        if (len(indices) != 1 or indices[0] + 1 >= len(left) - 1
                or left[indices[0] + 1].startswith("--")):
            raise ValueError("native preparation has incomplete or repeated input custody")
        value = Path(left[indices[0] + 1]).expanduser()
        return (value if value.is_absolute() else cwd / value).resolve()

    bundles = [left[index + 1] for index, value in enumerate(left[:-1]) if value == "--bundle"
               and index + 1 < len(left) - 1]
    names = [value.partition("=")[0] for value in bundles]
    if (not bundles or len(bundles) != left.count("--bundle") or len(names) != len(set(names))
            or any(not name or not value.partition("=")[2] for name, value in zip(names, bundles))):
        raise ValueError("native preparation source bundle inventory differs")
    paths = {"native": path("--directory"), "parent": path("--parent"), "bank": path("--bank"),
             "source": path("--source-report"), "folds": path("--folds")}
    command = [left[0], str(ROOT / "tools/train_semantic_native_program.py"), *left[2:-1]]
    for flag, name in (("--directory", "native"), ("--parent", "parent"), ("--bank", "bank"),
                       ("--source-report", "source"), ("--folds", "folds")):
        command[command.index(flag) + 1] = str(paths[name])
    normalized = []
    for index, value in enumerate(command):
        if value == "--bundle":
            name, _, raw = command[index + 1].partition("=")
            bundle_path = Path(raw).expanduser()
            command[index + 1] = name + "=" + str((bundle_path if bundle_path.is_absolute()
                                                  else cwd / bundle_path).resolve())
            normalized.append(command[index + 1])
    return {**paths, "command": command, "bundles": normalized}


def fit_jobs(paths: dict, directory: Path, *, python: str) -> list[dict]:
    from tools.run_semantic_native_micro_stages import job

    verify = [python, str(ROOT / "tools/verify_semantic_native_fit.py"),
        "--directory", str(paths["native"]), "--parent", str(paths["parent"]),
        "--bank", str(paths["bank"]), "--source-report", str(paths["source"]),
        "--output", str(directory / "fit-verification.json")]
    verify += [part for bundle in paths["bundles"] for part in ("--bundle", bundle)]
    return [job("native-fit", paths["command"], directory, timeout=15000.),
            job("native-fit-verify", verify, directory, timeout=3600.)]


def verify_preparation(preparation: Path, paths: dict) -> dict:
    from tools.evaluate_semantic_native_checkpoint import verified_document
    from tools.semantic_native_identifiability import verify_identifiability_preflight

    receipt = verified_preparation_document(preparation / "native-preparation.json")
    plan = verified_document(paths["native"] / "plan.json", "plan_sha256")
    supervision = verified_document(paths["native"] / "supervision.json")
    preflight = verify_identifiability_preflight(paths["native"], plan, supervision)
    if (receipt.get("schema") != "aura.semantic_source_native_preparation.v1"
            or receipt.get("plan_sha256") != plan["plan_sha256"]
            or receipt.get("supervision_receipt_sha256") != supervision["receipt_sha256"]
            or receipt.get("preflight_receipt_sha256") != preflight["receipt_sha256"]
            or preflight["strict_requirement"] is not True
            or receipt.get("model_weights_loaded") is not False
            or receipt.get("qualification_evidence") is not False
            or receipt.get("serving_authority") is not False):
        raise ValueError("native preparation receipt differs from its actual scoring inputs")
    for name, sha in plan["implementation"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != sha):
            raise ValueError("native fit handoff changed the prepared implementation")
    if (any(paths["native"].glob("checkpoint-*.json"))
            or (paths["native"] / "report.json").exists()):
        raise ValueError("native fit already has evidence; handoff cannot repeat training")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation-supervisor", type=Path, required=True)
    parser.add_argument("--preparation-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path)
    args = parser.parse_args()
    from tools.probe_semantic_proposer_crossfit import _digest, _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.run_detached_step import PLAN_FILE, _status, _verify_plan
    from tools.run_semantic_native_micro_stages import wait_for_fit

    preparation, directory = args.preparation_directory.resolve(), args.directory.resolve()
    configure_refit_environment(directory / "handoff.json")
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    supervisor = args.preparation_supervisor.resolve()
    supervised = json.loads((supervisor / PLAN_FILE).read_bytes())
    _verify_plan(supervised, supervisor / PLAN_FILE)
    command = supervised["command"]
    if (command[1:3] != ["-u", "tools/run_semantic_source_handoff.py"]
            or command.count("--directory") != 1
            or Path(command[command.index("--directory") + 1]).resolve() != preparation
            or "--wait-bank-supervisor" not in command):
        raise ValueError("native fit handoff belongs to another preparation supervisor")
    handoff = verified_preparation_document(preparation / "handoff.json")
    paths = preparation_paths(handoff)
    jobs = fit_jobs(paths, directory, python=sys.executable)
    body = {"schema": "aura.semantic_native_fit_handoff_plan.v1",
            "preparation_plan_sha256": supervised["plan_sha256"],
            "preparation_protocol_receipt_sha256": handoff["receipt_sha256"], "jobs": jobs,
            "qualification_evidence": False, "serving_authority": False}
    _save_if_absent(directory / "handoff.json", {**body, "receipt_sha256": _digest(body)})
    if args.policy_output is not None:
        _save_if_absent(args.policy_output, [{key: value for key, value in row.items()
                                           if key != "name"} for row in jobs])
        print(json.dumps({"stage": "policy_only", "commands": len(jobs)}), flush=True)
        return 0
    from core.runtime.detached_subprocess_broker import broker_available, run_brokered_process

    if not broker_available():
        raise ValueError("native fit handoff requires the existing detached supervisor broker")
    terminal = wait_for_fit(supervisor, timeout=50400.)
    if (_status(supervisor)["plan_sha256"] != supervised["plan_sha256"]
            or verified_preparation_document(preparation / "handoff.json") != handoff):
        raise ValueError("native preparation changed its supervised protocol")
    prepared = verify_preparation(preparation, paths)
    for row in jobs:
        print(json.dumps({"stage": row["name"], "status": "started"}), flush=True)
        outcome = run_brokered_process(row["command"], cwd=ROOT,
            stdout_path=Path(row["stdout_path"]), timeout_s=row["timeout_s_max"])
        if (outcome.returncode != 0 or outcome.status != "passed" or outcome.timed_out
                or not outcome.containment_verified):
            raise ValueError(f"native fit handoff command failed: {row['name']}:{outcome.status}")
        print(json.dumps({"stage": row["name"], "status": "process_completed",
                          "broker_receipt_sha256": outcome.receipt_sha256}), flush=True)
    from tools.adjudicate_semantic_native_micro_stages import verified_native_fit

    plan, checkpoint, verification = verified_native_fit(paths["native"], directory / "fit-verification.json")
    body = {"schema": "aura.semantic_native_fit_handoff_complete.v1",
            "preparation_terminal_receipt_sha256": terminal["receipt_sha256"],
            "preparation_receipt_sha256": prepared["receipt_sha256"],
            "training_plan_sha256": plan["plan_sha256"],
            "fit_verification_receipt_sha256": verification["receipt_sha256"],
            "selected_checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
            "learned_checkpoint_selected": checkpoint["step"] > 0,
            "semantic_correctness_measured": False, "general_transfer_proven": False,
            "qualification_evidence": False, "serving_authority": False}
    _save_if_absent(directory / "fit-completion.json", {**body, "receipt_sha256": _digest(body)})
    print(json.dumps({"stage": "native_fit_verified", **body}), flush=True)
    return 0 if body["learned_checkpoint_selected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
