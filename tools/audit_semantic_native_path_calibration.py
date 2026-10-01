#!/usr/bin/env python3
"""Replay source-calibration competitions across immutable native checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.learning.semantic_native_path_calibration import (  # noqa: E402
    native_path_profile,
    native_path_totals,
)
from tools.evaluate_semantic_native_checkpoint import (  # noqa: E402
    digest,
    selected_checkpoint,
    verified_document,
)


def calibration_competitions(training, supervision):
    """Require every calibration source and every alternative of its teacher path."""
    if (training.get("objective") not in {"grammar_choices", "grammar_source_pairs"}
            or supervision.get("plan_sha256") != training["plan_sha256"]
            or {row["source"] for row in supervision["rows"]}
                != set(training["captured_fit_ids"]) | set(training["calibration_ids"])):
        raise ValueError("path audit supervision differs from training")
    calibration = set(training["calibration_ids"])
    if (not calibration or len(calibration) != len(training["calibration_ids"])
            or calibration & (set(training["fit_ids"]) | set(training["held_ids"]))):
        raise ValueError("path audit calibration partition differs")
    grouped = {source: {} for source in sorted(calibration)}
    for row in supervision["rows"]:
        if row["source"] in calibration:
            grouped[row["source"]].setdefault(row["decision_index"], []).append(row)
    result = {}
    for source, decisions in grouped.items():
        if not decisions or sorted(decisions) != list(range(len(decisions))):
            raise ValueError("path audit teacher decisions are incomplete")
        result[source] = []
        for ordinal in sorted(decisions):
            rows = sorted(decisions[ordinal], key=lambda row: row["choice_index"])
            correct, kind = rows[0]["correct_index"], rows[0]["kind"]
            if ([row["choice_index"] for row in rows] != list(range(len(rows)))
                    or type(correct) is not int or not 0 <= correct < len(rows)
                    or any(row["correct_index"] != correct or row["kind"] != kind for row in rows)):
                raise ValueError("path audit teacher alternatives differ")
            result[source].append(rows)
        last = result[source][-1]
        if last[0]["kind"] != "termination" or last[last[0]["correct_index"]]["choice"] != "finish":
            raise ValueError("path audit teacher path lacks its exact finish")
    return result


def load_basis(directory):
    training, selected = selected_checkpoint(directory)
    report = verified_document(directory / "report.json")
    supervision = verified_document(directory / "supervision.json")
    if report["supervision_receipt_sha256"] != supervision["receipt_sha256"]:
        raise ValueError("path audit supervision receipt differs")
    competitions = calibration_competitions(training, supervision)
    checkpoints = [verified_document(directory / f"checkpoint-{step}.json")
                   for step in sorted(row["step"] for row in report["checkpoints"])]
    if checkpoints != sorted(report["checkpoints"], key=lambda row: row["step"]):
        raise ValueError("path audit checkpoint inventory differs")
    return training, selected, supervision, competitions, checkpoints


def verified_measurement(row, *, plan, checkpoint, source, competitions):
    if (row.get("plan_sha256") != plan["plan_sha256"] or row.get("source") != source
            or row.get("checkpoint_receipt_sha256") != checkpoint["receipt_sha256"]
            or len(row.get("decisions", [])) != len(competitions)):
        raise ValueError("path audit measurement identity differs")
    for measured, expected in zip(row["decisions"], competitions, strict=True):
        if (measured.get("kind") != expected[0]["kind"]
                or measured.get("correct_index") != expected[0]["correct_index"]
                or measured.get("choices") != [item["choice"] for item in expected]
                or len(measured.get("scores", [])) != len(expected)):
            raise ValueError("path audit measured alternatives differ")
    if row.get("profile") != native_path_profile(row["decisions"]):
        raise ValueError("path audit profile differs from measured scores")
    return row


def verify_audit(directory, training_directory):
    training, _selected, supervision, competitions, checkpoints = load_basis(training_directory)
    plan = verified_document(directory / "plan.json", "plan_sha256")
    report = verified_document(directory / "report.json")
    if (plan.get("schema") != "aura.native_path_calibration_plan.v1"
            or plan.get("training_plan_sha256") != training["plan_sha256"]
            or plan.get("supervision_receipt_sha256") != supervision["receipt_sha256"]
            or plan.get("sources") != sorted(competitions)
            or plan.get("checkpoints") != checkpoints
            or plan.get("held_labels_used") is not False
            or plan.get("serving_authority") is not False
            or report.get("plan_sha256") != plan["plan_sha256"]):
        raise ValueError("path audit source-only protocol differs")
    results = []
    for checkpoint in checkpoints:
        step = checkpoint["step"]
        rows = [verified_measurement(verified_document(directory / "rows" / f"{step}-{source}.json"),
                    plan=plan, checkpoint=checkpoint, source=source, competitions=competitions[source])
                for source in plan["sources"]]
        totals = native_path_totals(rows, plan["sources"])
        results.append({"step": step, "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
                        "totals": totals, "row_receipts": {row["source"]: row["receipt_sha256"] for row in rows}})
    if report.get("checkpoints") != results:
        raise ValueError("path audit reported totals differ")
    expected_files = {f"{checkpoint['step']}-{source}.json" for checkpoint in checkpoints for source in plan["sources"]}
    if {path.name for path in (directory / "rows").glob("*.json")} != expected_files:
        raise ValueError("path audit row inventory differs")
    return {"artifacts_verified": True, "report_receipt_sha256": report["receipt_sha256"],
            "checkpoints": [{"step": row["step"], "totals": row["totals"]} for row in results],
            "model_scores_recomputed": False, "training_selection_unchanged": True,
            "current_implementation_drift": sorted(name for name, sha in plan["implementation"].items()
                if not (ROOT / name).is_file() or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha),
            "general_transfer_proven": False, "serving_authority": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 3600 or args.verify and args.output is None:
        parser.error("path audit needs a finite bound and verification output")
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.directory / "report.json")
    if args.verify:
        body = {"schema": "aura.native_path_calibration_verification.v1",
                **verify_audit(args.directory, args.training_directory)}
        result = {**body, "receipt_sha256": digest(body)}
        _save_if_absent(args.output, result)
        print(json.dumps(result), flush=True)
        return
    training, selected, supervision, competitions, checkpoints = load_basis(args.training_directory)
    from tools.semantic_native_execution import execution_from_plan
    execution = execution_from_plan(training, check_installed=True)
    if execution is None or execution["precision"] != "float32":
        raise ValueError("path audit requires the measured training arithmetic")
    from core.brain.llm.model_registry import get_active_cortex_spec
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("path audit resident model differs from training")
    paths = ("tools/audit_semantic_native_path_calibration.py",
             "tools/semantic_native_adapters.py", "tools/semantic_native_adapter_layers.py",
             "core/learning/semantic_native_path_calibration.py", "tools/train_semantic_native_program.py",
             "core/learning/frozen_decoder_prefix.py", "core/learning/frozen_state_store.py",
             "tools/semantic_native_execution.py", "tools/semantic_native_prefix_reuse.py")
    body = {"schema": "aura.native_path_calibration_plan.v1", "training_plan_sha256": training["plan_sha256"],
            "supervision_receipt_sha256": supervision["receipt_sha256"], "checkpoints": checkpoints,
            "sources": sorted(competitions), "max_seconds": args.max_seconds,
            "original_selected_step": selected["step"], "partition": "source_calibration_only",
            "model_descriptor_sha256": spec.descriptor_sha256, "pointer_sha256": spec.pointer_sha256,
            "implementation": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths},
            "held_labels_used": False, "training_selection_unchanged": True,
            "serving_authority": False, "qualification_evidence": False}
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "sources": len(competitions),
                          "checkpoints": len(checkpoints), "plan_sha256": plan["plan_sha256"]}), flush=True)
        return
    import mlx.core as mx
    from mlx_lm import load
    from tools.semantic_native_adapters import install_native_adapters

    from core.learning.frozen_decoder_prefix import NativeDecoderSuffix
    from core.learning.semantic_native_program import NativeProgramSequence
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.semantic_native_execution import apply_execution
    from tools.semantic_native_prefix_reuse import open_reused_prefix
    from tools.train_semantic_native_program import native_loss

    if "reused_prefix_contract" not in training:
        raise ValueError("path audit requires an independently bound immutable prefix capture")
    states = open_reused_prefix(training["reused_prefix_contract"], training, supervision)
    started, results = time.monotonic(), []
    with (standalone_model_lane(owner_id=f"native-path-calibration:{args.directory.name}",
            model_path=str(spec.model_path), purpose="evaluation", require_exclusive=True,
            allow_owner_eviction=False, preemptible=False, metadata={"production_effect": False}),
          mlx_memory_envelope(fraction=.80)):
        model, _tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        mx.random.seed(training["seed"])
        install_native_adapters(model, training)
        apply_execution(model, training)
        suffix = NativeDecoderSuffix(model, split_at=len(model.layers) - training["suffix_layers"])
        for checkpoint in checkpoints:
            model.load_weights(str(args.training_directory / f"checkpoint-{checkpoint['step']}.safetensors"), strict=False)
            measured_rows = []
            for source, choices in competitions.items():
                path = args.directory / "rows" / f"{checkpoint['step']}-{source}.json"
                if path.exists():
                    measured_rows.append(verified_measurement(verified_document(path), plan=plan,
                        checkpoint=checkpoint, source=source, competitions=choices))
                    continue
                decisions = []
                for alternatives in choices:
                    scores = []
                    for row in alternatives:
                        if time.monotonic() - started > args.max_seconds:
                            raise TimeoutError("path audit bound reached; complete source receipts remain")
                        key = (source, row["decision_index"], row["choice_index"])
                        sequence = NativeProgramSequence(tuple(row["tokens"]), row["continuation_start"],
                                                         tuple(row["semantic_positions"]))
                        scores.append(-native_loss(suffix, states[key], sequence, summed=True,
                                                  scope="semantic_decisions").item())
                    decisions.append({"kind": alternatives[0]["kind"], "correct_index": alternatives[0]["correct_index"],
                                      "choices": [row["choice"] for row in alternatives], "scores": scores})
                body = {"schema": "aura.native_path_calibration_row.v1", "plan_sha256": plan["plan_sha256"],
                        "checkpoint_receipt_sha256": checkpoint["receipt_sha256"], "source": source,
                        "decisions": decisions, "profile": native_path_profile(decisions)}
                measured = {**body, "receipt_sha256": digest(body)}
                _save_if_absent(path, measured)
                measured_rows.append(measured)
                print(json.dumps({"stage": "source", "step": checkpoint["step"],
                    "observed": len(measured_rows), "population": len(competitions),
                    "exact_teacher_path": measured["profile"]["exact_teacher_path"]}), flush=True)
            results.append({"step": checkpoint["step"], "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
                            "totals": native_path_totals(measured_rows, plan["sources"]),
                            "row_receipts": {row["source"]: row["receipt_sha256"] for row in measured_rows}})
    body = {"schema": "aura.native_path_calibration.v1", "plan_sha256": plan["plan_sha256"],
            "checkpoints": results, "elapsed_seconds": time.monotonic() - started,
            "held_labels_used": False, "training_selection_unchanged": True, "serving_authority": False}
    _save_if_absent(args.directory / "report.json", {**body, "receipt_sha256": digest(body)})
    print(json.dumps({"stage": "complete", "elapsed_seconds": body["elapsed_seconds"],
                      "checkpoints": [{"step": row["step"], "totals": row["totals"]} for row in results]}), flush=True)


if __name__ == "__main__":
    main()
