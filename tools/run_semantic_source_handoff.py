#!/usr/bin/env python3
"""Continue a completed source fit into frozen folds and a disjoint source bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def source_fit_paths(plan: dict[str, Any]) -> dict[str, Any]:
    """Recover exact inputs and absent output locations from the supervised command."""
    command = plan["command"]
    cwd = Path(plan["cwd"])
    if (not isinstance(command, list) or len(command) < 3
            or command[1:3] != ["-u", "tools/train_compositional_semantic_sources.py"]
            or "--source-order-inputs" not in command):
        raise ValueError("source handoff requires the source-order compiler fit command")

    def values(flag: str) -> list[str]:
        indices = [index for index, value in enumerate(command) if value == flag]
        if any(index + 1 == len(command) or command[index + 1].startswith("--") for index in indices):
            raise ValueError("source fit command has an incomplete option")
        return [command[index + 1] for index in indices]

    def path(flag: str) -> Path:
        rows = values(flag)
        if len(rows) != 1:
            raise ValueError("source fit command has missing or repeated output custody")
        candidate = Path(rows[0]).expanduser()
        return (candidate if candidate.is_absolute() else cwd / candidate).resolve()

    if values("--binary-solver") != ["blocked_lbfgs"]:
        raise ValueError("source handoff requires the measured bounded solver")
    bundles = values("--bundle")
    names = [bundle.partition("=")[0] for bundle in bundles]
    if (not bundles or len(set(names)) != len(names)
            or any(not name or "=" not in bundle for name, bundle in zip(names, bundles, strict=True))):
        raise ValueError("source fit bundle inventory differs")
    normalized = []
    for bundle in bundles:
        name, _, raw = bundle.partition("=")
        if not raw:
            raise ValueError("source fit bundle has no path")
        bundle_path = Path(raw).expanduser()
        normalized.append(name + "=" + str((bundle_path if bundle_path.is_absolute()
                                              else cwd / bundle_path).resolve()))
    return {"candidate": path("--output"), "report": path("--report-output"),
            "checkpoints": path("--binary-checkpoints"), "bundles": normalized}


def handoff_jobs(paths: dict[str, Any], directory: Path, *, python: str) -> list[dict[str, Any]]:
    from tools.run_semantic_native_micro_stages import job

    bundles = [part for bundle in paths["bundles"] for part in ("--bundle", bundle)]
    folds = directory / "utterance-folds.json"
    preparation = directory / "fold-preparation.json"
    freeze = [python, str(ROOT / "tools/freeze_semantic_source_folds.py"),
              "--transducer", str(paths["candidate"]), "--source-report", str(paths["report"]),
              "--source-fit-only", "--preparation-receipt", str(preparation),
              "--output", str(folds), "--axis", "utterance", "--count", "3", "--seed", "0",
              *bundles]
    bank = directory / "bank"
    crossfit = [python, str(ROOT / "tools/probe_semantic_proposer_crossfit.py"),
                "--parent", str(paths["candidate"]), "--source-report", str(paths["report"]),
                "--folds", str(folds), "--fold", "0", "--directory", str(bank),
                "--max-charts", "4", "--max-graphs", "2", "--solve-seconds", "1",
                "--per-construction", "1", "--binary-solver", "blocked_lbfgs",
                "--binary-checkpoints", str(bank / "binary-checkpoints"), *bundles]
    return [job("freeze-source-folds", freeze, directory, timeout=1800.),
            job("fit-source-bank", crossfit, directory, timeout=14400.)]


def source_bank_directory(plan: dict[str, Any], source_fit_supervisor: Path) -> Path:
    command = plan["command"]
    if (not isinstance(command, list) or len(command) != 7
            or command[1:3] != ["-u", "tools/run_semantic_source_handoff.py"]
            or command[3] != "--source-fit-supervisor" or command[5] != "--directory"):
        raise ValueError("native preparation requires the supervised source-bank handoff")
    cwd = Path(plan["cwd"])

    def resolve(raw: str) -> Path:
        path = Path(raw).expanduser()
        return (path if path.is_absolute() else cwd / path).resolve()

    if resolve(command[4]) != source_fit_supervisor.resolve():
        raise ValueError("source bank belongs to another supervised fit")
    return resolve(command[6])


def native_preparation_jobs(paths: dict[str, Any], bank_root: Path, native: Path,
                            directory: Path, *, python: str, authority_key_file: Path | None = None) -> list[dict[str, Any]]:
    from tools.run_semantic_native_micro_stages import job

    bundles = [part for bundle in paths["bundles"] for part in ("--bundle", bundle)]
    command = [python, str(ROOT / "tools/train_semantic_native_program.py"),
        "--parent", str(paths["candidate"]), "--source-report", str(paths["report"]),
        "--folds", str(bank_root / "utterance-folds.json"), "--bank", str(bank_root / "bank"),
        "--directory", str(native), "--steps", "303", "--save-every", "101",
        "--rank", "8", "--layers", "1", "--precision", "float32", "--prefix-strategy", "trie",
        "--prefix-storage", "source_shards", "--prefix-resident-mib", "512",
        "--max-seconds", "14400", "--max-sequence-tokens", "1024",
        "--loss-scope", "semantic_decisions", "--objective", "grammar_source_pairs",
        "--register-encoding", "role_relative_v1", "--path-objective", "--joint-graph-contrasts", "4",
        "--source-pair-policy", "typed_choice_complete_v1",
        "--schedule-policy", "construction_depth_balanced_v1",
        "--calibration-per-construction", "1", "--held-per-construction", "1",
        "--require-identifiable-supervision", *bundles]
    if authority_key_file is not None:
        command += ["--authority-key-file", str(authority_key_file.expanduser().absolute())]
    return [job("native-plan", [*command, "--plan-only"], directory, timeout=1800.),
            job("native-supervision", [*command, "--supervision-only"], directory, timeout=3600.)]


def verify_native_launch_environment(command: list[str]) -> dict | None:
    """Check the child's declared arithmetic before freezing jobs or waiting."""
    from tools.semantic_native_execution import execution_contract

    def option(flag: str) -> str:
        indices = [index for index, value in enumerate(command) if value == flag]
        if (len(indices) != 1 or indices[0] + 1 == len(command)
                or command[indices[0] + 1].startswith("--")):
            raise ValueError("native launch requires explicit arithmetic options")
        return command[indices[0] + 1]

    return execution_contract(precision=option("--precision"),
                              prefix_strategy=option("--prefix-strategy"))


def verify_source_bank(bank_root: Path, paths: dict[str, Any], verification: dict) -> dict:
    from tools.train_nested_semantic_ranker import _verified_pair

    plan, report = _verified_pair(bank_root / "bank")
    if (plan.get("parent_receipt_sha256") != verification["candidate_receipt_sha256"]
            or plan.get("source_report_sha256") != verification["source_report_sha256"]
            or plan.get("folds_sha256") != hashlib.sha256(
                (bank_root / "utterance-folds.json").read_bytes()).hexdigest()
            or hashlib.sha256(paths["candidate"].read_bytes()).hexdigest() != verification["candidate_sha256"]
            or hashlib.sha256(paths["report"].read_bytes()).hexdigest() != verification["source_report_sha256"]):
        raise ValueError("source bank changed its candidate, partitions, or feature inputs")
    return report


def verify_source_fit_artifacts(paths: dict[str, Any]) -> dict[str, Any]:
    from core.learning.semantic_binary_fit_verification import verify_binary_fit_checkpoints
    from core.learning.semantic_fit_checkpoint import fit_identity
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict,
    )
    from tools.refit_semantic_argument_proposals import verify_source_report_identity

    candidate_raw, report_raw = paths["candidate"].read_bytes(), paths["report"].read_bytes()
    model = compositional_semantic_program_transducer_from_dict(json.loads(candidate_raw))
    report = json.loads(report_raw)
    verify_source_report_identity(report, model)
    if (report.get("fit_complete") is not True or report.get("evaluation_complete") is not False
            or report.get("test_examples_available_to_fit") != 0
            or report.get("binary_head_fit_checkpoints") != model.training_receipt.get(
                "binary_head_fit_checkpoints")):
        raise ValueError("source handoff has incomplete or inconsistent fitting evidence")
    binary = verify_binary_fit_checkpoints(model, paths["checkpoints"])
    body = {"schema": "aura.semantic_source_fit_handoff_verification.v1",
            "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
            "source_report_sha256": hashlib.sha256(report_raw).hexdigest(),
            "candidate_receipt_sha256": model.receipt_sha256,
            "binary_verification": binary, "validation_ids_sha256": report[
                "validation_example_ids_sha256"],
            "source_fit_complete": True, "qualification_evidence": False,
            "serving_authority": False}
    return {**body, "receipt_sha256": fit_identity(body)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-fit-supervisor", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path)
    parser.add_argument("--wait-bank-supervisor", type=Path)
    parser.add_argument("--native-directory", type=Path)
    parser.add_argument("--authority-key-file", type=Path)
    args = parser.parse_args()
    if (args.wait_bank_supervisor is None) != (args.native_directory is None):
        parser.error("native preparation needs both the source-bank supervisor and a fresh output directory")
    if args.authority_key_file is not None and args.native_directory is None:
        parser.error("cortex key custody belongs only to native preparation")
    from core.learning.semantic_fit_checkpoint import fit_identity
    from tools.probe_semantic_proposer_crossfit import _digest, _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.run_detached_step import PLAN_FILE, _status, _verify_plan
    from tools.run_semantic_native_micro_stages import wait_for_fit

    directory = args.directory.resolve()
    supervisor = args.source_fit_supervisor.resolve()
    configure_refit_environment(directory / "handoff.json")
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    source_plan_path = supervisor / PLAN_FILE
    source_plan = json.loads(source_plan_path.read_bytes())
    _verify_plan(source_plan, source_plan_path)
    paths = source_fit_paths(source_plan)
    bank_supervisor, bank_plan, bank_root = None, None, directory
    if args.wait_bank_supervisor is not None:
        bank_supervisor = args.wait_bank_supervisor.resolve()
        bank_plan_path = bank_supervisor / PLAN_FILE
        bank_plan = json.loads(bank_plan_path.read_bytes())
        _verify_plan(bank_plan, bank_plan_path)
        bank_root = source_bank_directory(bank_plan, supervisor)
    jobs = (handoff_jobs(paths, directory, python=sys.executable) if bank_supervisor is None else
            native_preparation_jobs(paths, bank_root, args.native_directory.resolve(), directory,
                                    python=sys.executable, authority_key_file=args.authority_key_file))
    if bank_supervisor is not None:
        verify_native_launch_environment(jobs[0]["command"])
    body = {"schema": "aura.semantic_source_handoff_plan.v1",
            "source_fit_plan_sha256": source_plan["plan_sha256"],
            "source_fit_supervisor": str(supervisor), "jobs": jobs,
            "qualification_evidence": False, "serving_authority": False}
    if bank_supervisor is not None:
        body.update(schema="aura.semantic_source_native_preparation_plan.v1",
                    source_bank_supervisor=str(bank_supervisor),
                    source_bank_plan_sha256=bank_plan["plan_sha256"])
    _save_if_absent(directory / "handoff.json", {**body, "receipt_sha256": fit_identity(body)})
    if args.policy_output is not None:
        _save_if_absent(args.policy_output, [{key: value for key, value in item.items()
                                           if key != "name"} for item in jobs])
        print(json.dumps({"stage": "policy_only", "commands": len(jobs)}), flush=True)
        return 0
    from core.runtime.detached_subprocess_broker import broker_available, run_brokered_process

    if not broker_available():
        raise ValueError("source handoff requires the existing detached supervisor broker")
    terminal = wait_for_fit(supervisor, timeout=14400.)
    status = _status(supervisor)
    if status["plan_sha256"] != source_plan["plan_sha256"]:
        raise ValueError("source fit supervisor changed its bound plan")
    verification = verify_source_fit_artifacts(paths)
    _save_if_absent(directory / "source-fit-verification.json", verification)
    print(json.dumps({"stage": "source_fit_verified", "terminal_receipt_sha256": terminal[
        "receipt_sha256"], "verification_sha256": verification["receipt_sha256"]}), flush=True)
    if bank_supervisor is not None:
        print(json.dumps({"stage": "source_bank_wait", "plan_sha256": bank_plan["plan_sha256"]}), flush=True)
        bank_terminal = wait_for_fit(bank_supervisor, timeout=30600.)
        if _status(bank_supervisor)["plan_sha256"] != bank_plan["plan_sha256"]:
            raise ValueError("source bank changed its supervised plan")
        verify_source_bank(bank_root, paths, verification)
    for item in jobs:
        print(json.dumps({"stage": item["name"], "status": "started"}), flush=True)
        result = run_brokered_process(item["command"], cwd=ROOT,
            stdout_path=Path(item["stdout_path"]), timeout_s=item["timeout_s_max"])
        if (result.returncode != 0 or result.status != "passed" or result.timed_out
                or not result.containment_verified):
            raise ValueError(f"source handoff command failed: {item['name']}:{result.status}")
        print(json.dumps({"stage": item["name"], "status": "process_completed",
                          "broker_receipt_sha256": result.receipt_sha256}), flush=True)
    if bank_supervisor is not None:
        from tools.evaluate_semantic_native_checkpoint import verified_document
        from tools.semantic_native_identifiability import verify_identifiability_preflight

        native = args.native_directory.resolve()
        plan = verified_document(native / "plan.json", "plan_sha256")
        supervision = verified_document(native / "supervision.json")
        proof = verify_identifiability_preflight(native, plan, supervision)
        completed = {"schema": "aura.semantic_source_native_preparation.v1",
            "source_fit_terminal_receipt_sha256": terminal["receipt_sha256"],
            "source_bank_terminal_receipt_sha256": bank_terminal["receipt_sha256"],
            "source_fit_verification_sha256": verification["receipt_sha256"],
            "plan_sha256": plan["plan_sha256"],
            "supervision_receipt_sha256": supervision["receipt_sha256"],
            "preflight_receipt_sha256": proof["receipt_sha256"],
            "model_weights_loaded": False, "qualification_evidence": False, "serving_authority": False}
        completed["receipt_sha256"] = fit_identity(completed)
        _save_if_absent(directory / "native-preparation.json", completed)
        print(json.dumps({"stage": "native_preparation_complete", **completed}), flush=True)
        return 0
    bank = directory / "bank"
    plan, report = (json.loads((bank / name).read_bytes()) for name in ("plan.json", "report.json"))
    for document, field in ((plan, "plan_sha256"), (report, "receipt_sha256")):
        if document.get(field) != _digest({key: value for key, value in document.items() if key != field}):
            raise ValueError("source bank final artifact identity differs")
    if (report.get("plan_sha256") != plan["plan_sha256"]
            or set(report.get("row_receipts", {})) != set(plan["held_ids"])):
        raise ValueError("source bank is incomplete; no native handoff is permitted")
    print(json.dumps({"stage": "source_bank_complete", "receipt_sha256": report["receipt_sha256"],
                      "qualification_evidence": False, "serving_authority": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
