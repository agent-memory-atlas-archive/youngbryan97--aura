#!/usr/bin/env python3
"""Measure a fitted LoRA residual on complete source-calibration grammar paths."""

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

from core.learning.semantic_native_path_calibration import (  # noqa: E402
    native_path_profile,
    native_path_totals,
)
from tools.audit_semantic_native_path_calibration import calibration_competitions  # noqa: E402
from tools.evaluate_semantic_native_checkpoint import (  # noqa: E402
    digest,
    selected_checkpoint,
    verified_document,
)


def residual_scales(values: list[float]) -> tuple[float, ...]:
    if (len(values) < 3 or any(type(value) is not float or not math.isfinite(value)
                               or not 0. <= value <= 1. for value in values)
            or sorted(set(values)) != values or values[0] != 0. or values[-1] != 1.):
        raise ValueError("residual calibration needs ordered 0, interior scale, and 1 endpoints")
    return tuple(values)


def residual_training_contract(training: dict, selected: dict) -> None:
    """Admit only completed, rejected path fits with the same calibration basis."""
    if (training.get("schema") not in {"aura.semantic_native_fit_plan.v5",
                                      "aura.semantic_native_fit_plan.v6"}
            or selected.get("step") != 0 or training.get("suffix_layers") != 1
            or "reused_prefix_contract" not in training):
        raise ValueError("native residual requires a measured, rejected path suffix fit")
    if training["schema"].endswith(".v6"):
        from core.learning.semantic_native_path_objective import GRAMMAR_PATH_CONTRACT
        from core.learning.semantic_native_typed_source_pairs import TYPED_SOURCE_PAIR_CONTRACT

        if (training.get("grammar_source_pair_contract") != TYPED_SOURCE_PAIR_CONTRACT
                or training.get("grammar_path_objective_contract") != GRAMMAR_PATH_CONTRACT
                or training.get("contrast_policy") != "all_native_type_admitted_teacher_decisions_v1"):
            raise ValueError("native residual v6 source contrast contract differs")


def residual_admission(rows_by_scale: dict[float, list[dict]], sources: list[str]) -> dict:
    """Select from source paths only; a lower loss cannot excuse a lost path."""
    if not rows_by_scale or 0. not in rows_by_scale:
        raise ValueError("residual admission needs an unfitted source baseline")
    totals = {scale: native_path_totals(rows, sources) for scale, rows in rows_by_scale.items()}
    exact = {scale: {row["source"] for row in rows
                     if native_path_profile(row["decisions"])["exact_teacher_path"]}
             for scale, rows in rows_by_scale.items()}
    baseline = exact[0.]
    adjudication = [{"scale": scale, "exact_teacher_paths": totals[scale]["exact_teacher_paths"],
                     "lost_baseline_sources": sorted(baseline - exact[scale]),
                     "new_exact_sources": sorted(exact[scale] - baseline),
                     "eligible": baseline <= exact[scale]}
                    for scale in sorted(rows_by_scale)]
    selected = min((row for row in adjudication if row["eligible"]),
                   key=lambda row: (-row["exact_teacher_paths"], row["scale"]))
    return {"selected_scale": selected["scale"], "adjudication": adjudication,
            "totals": {str(scale): totals[scale] for scale in sorted(totals)}}


def native_lora_sites(model, training) -> tuple:
    from mlx_lm.tuner.lora import LoRALinear

    sites = []
    for layer in model.layers[-training["suffix_layers"]:]:
        for name in training["adapter_keys"]:
            parent, target = name.split(".")
            module = getattr(getattr(layer, parent, None), target, None)
            if module is not None:
                if not isinstance(module, LoRALinear) or module.scale != 16.:
                    raise ValueError("native residual adapter geometry differs")
                sites.append(module)
    if not sites:
        raise ValueError("native residual has no measured LoRA sites")
    return tuple(sites)


def measured_row(path: Path, *, plan: dict, source: str, scale: float,
                 choices: list[list[dict]]) -> dict:
    row = verified_document(path)
    if (row.get("schema") != "aura.native_residual_calibration_row.v1"
            or row.get("plan_sha256") != plan["plan_sha256"] or row.get("source") != source
            or row.get("scale") != scale
            or len(row.get("decisions", [])) != len(choices)):
        raise ValueError("native residual row identity differs")
    for decision, alternatives in zip(row["decisions"], choices, strict=True):
        if (decision["kind"] != alternatives[0]["kind"]
                or decision["correct_index"] != alternatives[0]["correct_index"]
                or decision["choices"] != [item["choice"] for item in alternatives]
                or len(decision["scores"]) != len(alternatives)):
            raise ValueError("native residual alternatives differ")
    if row["profile"] != native_path_profile(row["decisions"]):
        raise ValueError("native residual path profile differs")
    return row


def run(args) -> None:
    from core.brain.llm.model_registry import get_active_cortex_spec
    from core.learning.semantic_native_program import NativeProgramSequence
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment
    from tools.semantic_native_execution import apply_execution, execution_from_plan
    from tools.semantic_native_prefix_reuse import open_reused_prefix
    from tools.train_semantic_native_program import native_loss

    configure_refit_environment(args.directory / "report.json")
    scales = residual_scales(args.scale)
    training, selected = selected_checkpoint(args.training_directory)
    residual_training_contract(training, selected)
    candidate = verified_document(args.training_directory / f"checkpoint-{args.candidate_step}.json")
    if (candidate["step"] == 0 or candidate["plan_sha256"] != training["plan_sha256"]
            or candidate["step"] not in {row["step"] for row in
                                      verified_document(args.training_directory / "report.json")["checkpoints"]}):
        raise ValueError("native residual candidate is not a measured fitted checkpoint")
    supervision = verified_document(args.training_directory / "supervision.json")
    competitions = calibration_competitions(training, supervision)
    sources = sorted(competitions)
    endpoint_rows = {
        0.: {row["source"]: row for row in
             verified_document(args.training_directory / "calibration-paths-0.json")["rows"]},
        1.: {row["source"]: row for row in
             verified_document(args.training_directory /
                               f"calibration-paths-{args.candidate_step}.json")["rows"]},
    }
    if any(set(rows) != set(sources) for rows in endpoint_rows.values()):
        raise ValueError("native residual endpoint population differs")
    execution = execution_from_plan(training, check_installed=True)
    if execution is None or execution["precision"] != "float32":
        raise ValueError("native residual requires the measured FP32 arithmetic")
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("native residual checkpoint identity differs")
    paths = ("tools/calibrate_semantic_native_residual.py",
             "tools/audit_semantic_native_path_calibration.py",
             "tools/train_semantic_native_program.py",
             "core/learning/frozen_state_store.py",
             "core/learning/frozen_decoder_prefix.py",
             "core/learning/semantic_native_path_calibration.py",
             "tools/semantic_native_prefix_reuse.py")
    if training["schema"].endswith(".v6"):
        paths += ("core/learning/semantic_native_typed_source_pairs.py",)
    body = {"schema": "aura.native_residual_calibration_plan.v1",
            "training_plan_sha256": training["plan_sha256"],
            "training_report_receipt_sha256": verified_document(
                args.training_directory / "report.json")["receipt_sha256"],
            "baseline_checkpoint_receipt_sha256": selected["receipt_sha256"],
            "candidate_checkpoint_receipt_sha256": candidate["receipt_sha256"],
            "supervision_receipt_sha256": supervision["receipt_sha256"],
            "scales": scales, "sources": sources, "max_seconds": args.max_seconds,
            "model_descriptor_sha256": spec.descriptor_sha256,
            "pointer_sha256": spec.pointer_sha256,
            "implementation": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in paths},
            "held_labels_used": False, "training_selection_unchanged": True,
            "serving_authority": False, "qualification_evidence": False}
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "plan_sha256": plan["plan_sha256"],
                          "sources": len(sources), "scales": scales}), flush=True)
        return

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.frozen_decoder_prefix import NativeDecoderSuffix

    states = open_reused_prefix(training["reused_prefix_contract"], training, supervision)
    started = time.monotonic()
    results = {}
    with (standalone_model_lane(owner_id=f"native-residual:{args.directory.name}",
            model_path=str(spec.model_path), purpose="evaluation", require_exclusive=True,
            allow_owner_eviction=False, preemptible=False, metadata={"production_effect": False}),
          mlx_memory_envelope(fraction=.80)):
        model, _tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        mx.random.seed(training["seed"])
        linear_to_lora_layers(model, training["suffix_layers"], {
            "rank": training["rank"], "scale": 16., "dropout": 0.,
            "keys": training["adapter_keys"]})
        apply_execution(model, training)
        sites = native_lora_sites(model, training)
        model.load_weights(str(args.training_directory /
                               f"checkpoint-{args.candidate_step}.safetensors"), strict=False)
        suffix = NativeDecoderSuffix(model, split_at=len(model.layers) - training["suffix_layers"])
        for scale in scales:
            for site in sites:
                site.scale = 16. * scale
            measured = []
            for source in sources:
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("native residual calibration exceeded its finite bound")
                path = args.directory / "rows" / f"{scale:g}-{source}.json"
                if path.exists():
                    row = measured_row(path, plan=plan, source=source, scale=scale,
                                       choices=competitions[source])
                else:
                    decisions = []
                    for alternatives in competitions[source]:
                        scores = []
                        for choice in alternatives:
                            key = (source, choice["decision_index"], choice["choice_index"])
                            sequence = NativeProgramSequence(tuple(choice["tokens"]),
                                choice["continuation_start"], tuple(choice["semantic_positions"]))
                            scores.append(-native_loss(suffix, states[key], sequence, summed=True,
                                                       scope="semantic_decisions").item())
                        decisions.append({"kind": alternatives[0]["kind"],
                            "correct_index": alternatives[0]["correct_index"],
                            "choices": [choice["choice"] for choice in alternatives],
                            "scores": scores})
                    row_body = {"schema": "aura.native_residual_calibration_row.v1",
                                "plan_sha256": plan["plan_sha256"], "source": source,
                                "scale": scale, "decisions": decisions,
                                "profile": native_path_profile(decisions)}
                    row = {**row_body, "receipt_sha256": digest(row_body)}
                    _save_if_absent(path, row)
                if scale in endpoint_rows:
                    reference = endpoint_rows[scale][source]
                    errors = [abs(actual - expected)
                              for decision, prior in zip(row["decisions"], reference["decisions"], strict=True)
                              for actual, expected in zip(decision["scores"], prior["scores"], strict=True)]
                    if (max(errors, default=0.) > 0.015625
                            or row["profile"] != native_path_profile(reference["decisions"])):
                        raise ValueError("native residual endpoint does not reproduce measured checkpoint")
                measured.append(row)
                print(json.dumps({"stage": "source", "scale": scale,
                                  "observed": len(measured), "population": len(sources)}), flush=True)
            results[scale] = measured
    admission = residual_admission(results, sources)
    report_body = {"schema": "aura.native_residual_calibration.v1",
                   "plan_sha256": plan["plan_sha256"], **admission,
                   "row_receipts": {str(scale): {row["source"]: row["receipt_sha256"] for row in rows}
                                    for scale, rows in results.items()},
                   "elapsed_seconds": time.monotonic() - started,
                   "held_labels_used": False, "training_selection_unchanged": True,
                   "serving_authority": False, "general_transfer_proven": False}
    _save_if_absent(args.directory / "report.json",
                    {**report_body, "receipt_sha256": digest(report_body)})
    print(json.dumps({"stage": "complete", "selected_scale": admission["selected_scale"],
                      "totals": admission["totals"]}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--candidate-step", type=int, required=True)
    parser.add_argument("--scale", type=float, action="append", required=True)
    parser.add_argument("--max-seconds", type=float, default=3600.)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 0. < args.max_seconds <= 7200.:
        parser.error("native residual needs a finite bound")
    run(args)


if __name__ == "__main__":
    main()
