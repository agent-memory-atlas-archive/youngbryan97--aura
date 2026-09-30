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
    args = parser.parse_args()
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
    jobs = handoff_jobs(paths, directory, python=sys.executable)
    body = {"schema": "aura.semantic_source_handoff_plan.v1",
            "source_fit_plan_sha256": source_plan["plan_sha256"],
            "source_fit_supervisor": str(supervisor), "jobs": jobs,
            "qualification_evidence": False, "serving_authority": False}
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
    for item in jobs:
        print(json.dumps({"stage": item["name"], "status": "started"}), flush=True)
        result = run_brokered_process(item["command"], cwd=ROOT,
            stdout_path=Path(item["stdout_path"]), timeout_s=item["timeout_s_max"])
        if (result.returncode != 0 or result.status != "passed" or result.timed_out
                or not result.containment_verified):
            raise ValueError(f"source handoff command failed: {item['name']}:{result.status}")
        print(json.dumps({"stage": item["name"], "status": "process_completed",
                          "broker_receipt_sha256": result.receipt_sha256}), flush=True)
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
