#!/usr/bin/env python3
"""Measure frozen source reuse against single-row full-sequence inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def precision_contract(mode):
    """Changing arithmetic creates a new experiment, without a larger allowance."""
    if mode == "native":
        return None
    if mode not in {"float32", "float16"} or os.environ.get("MLX_ENABLE_TF32") != "0":
        raise ValueError("precision probe requires MLX_ENABLE_TF32=0 at process launch")
    body = {"schema": "aura.native_branch_precision.v1", "mode": mode,
            "floating_parameters": "float32", "packed_integer_weights": "unchanged",
            "reduced_precision_matmul": False, "target_logprob_tolerance": .015625,
            "tolerance_basis": "unchanged_native_bfloat16_probe_absolute_allowance",
            "native_full_sequence_reference_required": True,
            "native_ranking_change_grants_no_equivalence": True}
    if mode == "float16":
        body.update(schema="aura.native_branch_precision.v2",
                    floating_parameters="bfloat16_to_float16_other_dtypes_unchanged",
                    native_float32_parameters="same_objects",
                    recurrent_accumulator="installed_float32_state_unchanged",
                    nonfinite_parameter_cast="refuse",
                    nonfinite_hidden_or_target_logprob="refuse")
    return body


def float32_parameters(model):
    """Cast floating parameters and prove packed weights retain their objects."""
    return convert_parameter_precision(model, mode="float32")


def convert_parameter_precision(model, *, mode):
    """Keep native FP32 state and packed weights when testing a narrower dtype."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    before = dict(tree_flatten(model.parameters()))
    if not before:
        raise ValueError("precision probe needs declared model parameters")
    if mode == "float32":
        model.set_dtype(mx.float32)
    elif mode == "float16":
        if not any(value.dtype == mx.bfloat16 for value in before.values()):
            raise ValueError("float16 probe needs native bfloat16 parameters")
        model.set_dtype(mx.float16, predicate=lambda dtype: dtype == mx.bfloat16)
    else:
        raise ValueError("unsupported parameter precision")
    after = dict(tree_flatten(model.parameters()))
    if before.keys() != after.keys():
        raise ValueError("precision conversion changed the parameter inventory")
    for name, value in before.items():
        updated = after[name]
        if value.shape != updated.shape:
            raise ValueError("precision conversion changed parameter shape")
        if mode == "float16" and value.dtype != mx.bfloat16:
            if updated is not value:
                raise ValueError("float16 conversion replaced a protected native parameter")
        elif mx.issubdtype(value.dtype, mx.floating):
            if updated.dtype != (mx.float32 if mode == "float32" else mx.float16):
                raise ValueError("precision conversion left a low precision parameter")
        elif updated is not value:
            raise ValueError("precision conversion replaced a packed integer parameter")
    floating = [value for value in after.values() if mx.issubdtype(value.dtype, mx.floating)]
    if floating and not mx.all(mx.stack([mx.all(mx.isfinite(value)) for value in floating])).item():
        raise ValueError("precision conversion produced nonfinite parameters")
    body = {"schema": "aura.native_parameter_precision.v1",
            "before_dtype_counts": dict(Counter(str(value.dtype) for value in before.values())),
            "after_dtype_counts": dict(Counter(str(value.dtype) for value in after.values())),
            "before_parameter_bytes": sum(value.nbytes for value in before.values()),
            "after_parameter_bytes": sum(value.nbytes for value in after.values()),
            "unchanged_nonfloating_parameters": sum(not mx.issubdtype(value.dtype, mx.floating)
                                                      for value in before.values())}
    if mode == "float16":
        body.update(schema="aura.native_parameter_precision.v2", mode=mode,
                    converted_bfloat16_parameters=sum(value.dtype == mx.bfloat16 for value in before.values()),
                    unchanged_float32_parameters=sum(value.dtype == mx.float32 for value in before.values()),
                    parameters_finite=True)
    return body


def finite_target_values(logprobs):
    """Persist every compared target so numerical accounting can be replayed."""
    values = logprobs.tolist()
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("precision probe produced nonfinite target log probabilities")
    return values


def installed_arithmetic_basis():
    """Pin the installed kernels whose arithmetic this probe compares."""
    files = ("mlx_lm/models/base.py", "mlx_lm/models/gated_delta.py",
             "mlx_lm/models/qwen3_5.py", "mlx_lm/models/qwen3_next.py")
    distribution = importlib.metadata.distribution("mlx-lm")
    return {"mlx_version": importlib.metadata.version("mlx"),
            "mlx_lm_version": distribution.version,
            "implementation": {name: hashlib.sha256(distribution.locate_file(name).read_bytes()).hexdigest()
                               for name in files},
            "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32")}


def target_logprobs(logits, targets):
    import mlx.core as mx

    values = logits[0].astype(mx.float32)
    return mx.take_along_axis(values, targets[:, None], axis=1)[:, 0] - mx.logsumexp(values, axis=-1)


def ranked_score_equivalence(reference, branch, *, tolerance):
    """A numerical allowance cannot authorize a different winning alternative."""
    if (not reference or len(reference) != len(branch) or not math.isfinite(tolerance)
            or tolerance < 0 or any(not math.isfinite(value) for value in (*reference, *branch))):
        raise ValueError("branch score comparison needs finite matched alternatives")
    reference_order = sorted(range(len(reference)), key=lambda index: (-reference[index], index))
    branch_order = sorted(range(len(branch)), key=lambda index: (-branch[index], index))
    error = max(abs(left - right) for left, right in zip(reference, branch, strict=True))
    return {"max_score_error": error, "score_tolerance": tolerance,
            "winner_preserved": reference_order[0] == branch_order[0],
            "complete_ranking_preserved": reference_order == branch_order,
            "accepted": error <= tolerance and reference_order == branch_order}


def branch_source_groups(supervision, *, count):
    from tools.probe_semantic_native_prefix_grouping import supervision_sequences

    if type(count) is not int or not 1 <= count <= 16:
        raise ValueError("branch probe needs a finite source population")
    groups = {}
    for key, sequence in supervision_sequences(supervision).items():
        groups.setdefault(key[0], {})[key[1:]] = sequence
    eligible = [(source, tuple(group[key] for key in sorted(group)))
                for source, group in sorted(groups.items()) if len(group) >= 2]
    if len(eligible) < count:
        raise ValueError("branch probe lacks complete source-bound alternatives")
    return tuple(eligible[:count])


def branch_choice_partitions(supervision, source):
    """Compare rankings within real competitions, never across decision steps."""
    rows = [row for row in supervision["rows"] if row["source"] == source]
    if "grammar_choice_contract" not in supervision:
        return ({"kind": "whole_program", "indices": tuple(range(len(rows)))},)
    rows.sort(key=lambda row: (row["decision_index"], row["choice_index"]))
    decisions = {}
    for index, row in enumerate(rows):
        decisions.setdefault(row["decision_index"], []).append((index, row))
    if sorted(decisions) != list(range(len(decisions))):
        raise ValueError("branch probe grammar has missing decision groups")
    result = []
    for ordinal, entries in decisions.items():
        if ([row["choice_index"] for _index, row in entries] != list(range(len(entries)))
                or len({row["kind"] for _index, row in entries}) != 1
                or len({row["correct_index"] for _index, row in entries}) != 1
                or not 0 <= entries[0][1]["correct_index"] < len(entries)):
            raise ValueError("branch probe grammar choices are incomplete or inconsistent")
        result.append({"decision_index": ordinal, "kind": entries[0][1]["kind"],
                       "indices": tuple(index for index, _row in entries)})
    return tuple(result)


def partitioned_score_equivalence(reference, branch, partitions, *, tolerance):
    return tuple({**partition, "comparison": ranked_score_equivalence(
        [reference[index] for index in partition["indices"]],
        [branch[index] for index in partition["indices"]], tolerance=tolerance)}
        for partition in partitions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--precision", choices=("native", "float32", "float16"), default="native")
    parser.add_argument("--branch-strategy", choices=("anchor", "trie"), default="anchor")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.sources <= 16 or not 0 < args.max_seconds <= 3600:
        parser.error("branch probe needs bounded sources and runtime")
    if args.branch_strategy == "trie" and os.environ.get("MLX_ENABLE_TF32") != "0":
        parser.error("trie arithmetic probe requires MLX_ENABLE_TF32=0 at process launch")
    precision = precision_contract(args.precision)
    from tools.evaluate_semantic_native_checkpoint import (
        digest,
        selected_checkpoint,
        verified_document,
    )
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.directory / "report.json")
    training, selected = selected_checkpoint(args.training_directory)
    supervision = verified_document(args.training_directory / "supervision.json")
    if supervision["plan_sha256"] != training["plan_sha256"]:
        raise ValueError("branch probe supervision differs from training")
    groups = branch_source_groups(supervision, count=args.sources)
    grammar = "grammar_choice_contract" in supervision
    partitions = {source: branch_choice_partitions(supervision, source) for source, _sequences in groups}
    paths = ("tools/probe_semantic_native_prefix_branches.py", "core/learning/frozen_prefix_branches.py",
             "core/learning/frozen_decoder_prefix.py", "core/brain/llm/decoder_topology.py",
             "tools/probe_semantic_native_prefix_grouping.py", "tools/train_semantic_native_program.py")
    implementation = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}
    from core.brain.llm.model_registry import get_active_cortex_spec
    spec = get_active_cortex_spec(force_refresh=True)
    if (spec is None or not spec.exact_identity
            or spec.descriptor_sha256 != training["model_descriptor_sha256"]
            or spec.pointer_sha256 != training["pointer_sha256"]
            or spec.model_path.resolve() != Path(training["model_path"]).resolve()):
        raise ValueError("branch probe model differs from training")
    body = {"schema": "aura.native_prefix_branch_probe_plan.v1",
            "training_plan_sha256": training["plan_sha256"],
            "checkpoint_receipt_sha256": selected["receipt_sha256"],
            "supervision_receipt_sha256": supervision["receipt_sha256"],
            "model_descriptor_sha256": spec.descriptor_sha256, "pointer_sha256": spec.pointer_sha256,
            "sources": [source for source, _ in groups], "implementation": implementation,
            "max_seconds": args.max_seconds, "tolerance_basis": "two_logit_dtype_eps_per_target",
            "model_lane_requirement": "exclusive_process_lease_no_owner_eviction",
            "ranking_must_be_identical": True, "serving_authority": False,
            "qualification_evidence": False}
    if precision is not None:
        body.update(schema="aura.native_prefix_branch_probe_plan.v2",
                    precision_contract=precision, installed_arithmetic=installed_arithmetic_basis(),
                    tolerance_basis=precision["tolerance_basis"])
    if args.precision == "float16":
        body.update(schema="aura.native_prefix_branch_probe_plan.v3",
                    raw_target_logprob_arrays_required=True)
    if grammar or args.branch_strategy == "trie":
        body.update(schema="aura.native_prefix_branch_probe_plan.v4",
                    branch_strategy=args.branch_strategy, choice_partitions=partitions,
                    raw_target_logprob_arrays_required=True,
                    ranking_scope="within_each_declared_inference_competition")
        body["installed_arithmetic"] = installed_arithmetic_basis()
    plan = {**body, "plan_sha256": digest(body)}
    _save_if_absent(args.directory / "plan.json", plan)
    if args.plan_only:
        print(json.dumps({"stage": "plan_only", "plan_sha256": plan["plan_sha256"]}), flush=True)
        return
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.frozen_prefix_branches import FrozenPrefixBranches, native_source_anchor
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.train_semantic_native_program import native_prediction_positions

    started, rows = time.monotonic(), []
    with (standalone_model_lane(owner_id=f"native-branch-probe:{args.directory.name}",
            model_path=str(spec.model_path), purpose="evaluation", preemptible=False,
            require_exclusive=True, allow_owner_eviction=False,
            metadata={"production_effect": False}), mlx_memory_envelope(fraction=.8)):
        model, _ = load(str(spec.model_path))
        model.freeze()
        model.eval()
        split = len(model.layers) - training["suffix_layers"]
        prefix, suffix = FrozenDecoderPrefix(model, split_at=split), NativeDecoderSuffix(model, split_at=split)
        mx.random.seed(training["seed"])
        linear_to_lora_layers(model, training["suffix_layers"], {
            "rank": training["rank"], "scale": 16., "dropout": 0., "keys": training["adapter_keys"]})
        model.load_weights(str(args.training_directory / f"checkpoint-{selected['step']}.safetensors"), strict=False)
        native_scores, precision_receipt = {}, None
        if precision is not None:
            for source, sequences in groups:
                values = []
                for sequence in sequences:
                    if time.monotonic() - started > args.max_seconds:
                        raise TimeoutError("branch probe reached its finite runtime bound")
                    positions = tuple(index - 1 for index in native_prediction_positions(sequence,
                        scope=training["loss_scope"]))
                    targets = mx.array([sequence.tokens[index + 1] for index in positions], dtype=mx.int32)
                    hidden = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                    logits = suffix(hidden, logit_positions=positions)
                    logprobs = target_logprobs(logits, targets)
                    values.append({"logprobs": finite_target_values(logprobs), "score": float(mx.sum(logprobs).item()),
                                   "logit_dtype": str(logits.dtype)})
                    if len(values) % 16 == 0:
                        print(json.dumps({"stage": "native_precision_progress", "source": source,
                                          "captured": len(values), "population": len(sequences)}), flush=True)
                native_scores[source] = values
                print(json.dumps({"stage": "native_precision_reference", "source": source,
                                  "alternatives": len(values)}), flush=True)
            del hidden, logits, logprobs
            precision_receipt = convert_parameter_precision(model, mode=args.precision)
            mx.eval(model.parameters())
        for source, sequences in groups:
            begin = time.monotonic()
            branches = FrozenPrefixBranches(model, split_at=split,
                anchor_tokens=native_source_anchor(sequences), max_tokens=training["max_sequence_tokens"])
            branch_seconds, full_seconds = time.monotonic() - begin, 0.
            cached_states = None
            if args.branch_strategy == "trie":
                begin = time.monotonic()
                cached_states = branches.capture_many(tuple(tuple(sequence.tokens[:-1]) for sequence in sequences))
                branch_seconds += time.monotonic() - begin
            reference_scores, branch_scores, errors, score_tolerance = [], [], [], 0.
            native_precision_errors = []
            reference_targets, branch_targets = [], []
            for sequence_index, sequence in enumerate(sequences):
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("branch probe reached its finite runtime bound")
                positions = tuple(index - 1 for index in native_prediction_positions(sequence,
                    scope=training["loss_scope"]))
                targets = mx.array([sequence.tokens[index + 1] for index in positions], dtype=mx.int32)
                begin = time.monotonic()
                full = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                expected_dtype = mx.float16 if args.precision == "float16" else mx.float32
                if precision is not None and (full.dtype != expected_dtype
                                              or not mx.all(mx.isfinite(full)).item()):
                    raise ValueError("precision parameters did not produce finite declared prefix states")
                reference = suffix(full, logit_positions=positions)
                mx.eval(reference)
                full_seconds += time.monotonic() - begin
                begin = time.monotonic()
                cached = (cached_states[sequence_index] if cached_states is not None else
                          branches.capture(tuple(sequence.tokens[:-1])))
                alternate = suffix(cached, logit_positions=positions)
                mx.eval(alternate)
                branch_seconds += time.monotonic() - begin
                left, right = target_logprobs(reference, targets), target_logprobs(alternate, targets)
                reference_targets.append(finite_target_values(left))
                branch_targets.append(finite_target_values(right))
                tolerance = (precision["target_logprob_tolerance"] if precision is not None else
                             2 * max(mx.finfo(reference.dtype).eps, mx.finfo(alternate.dtype).eps))
                errors.append(float(mx.max(mx.abs(left - right)).item()))
                score_tolerance = max(score_tolerance, tolerance * len(positions))
                reference_scores.append(float(mx.sum(left).item()))
                branch_scores.append(float(mx.sum(right).item()))
                if precision is not None:
                    native_precision_errors.append(max(abs(a - b) for a, b in zip(
                        native_scores[source][sequence_index]["logprobs"], left.tolist(), strict=True)))
                if (sequence_index + 1) % 16 == 0:
                    print(json.dumps({"stage": "branch_comparison_progress", "source": source,
                                      "compared": sequence_index + 1, "population": len(sequences)}), flush=True)
            comparison = ranked_score_equivalence(reference_scores, branch_scores, tolerance=score_tolerance)
            choice_comparisons = partitioned_score_equivalence(reference_scores, branch_scores,
                partitions[source], tolerance=score_tolerance)
            row = {"source": source, "alternatives": len(sequences), "reference_scores": reference_scores,
                   "branch_scores": branch_scores, "target_logprob_errors": errors,
                   "target_tolerance": tolerance, "comparison": comparison,
                   "accepted": all(value["comparison"]["accepted"] for value in choice_comparisons)
                               and max(errors) <= tolerance,
                   "full_seconds": full_seconds, "branch_seconds": branch_seconds,
                   "cache": branches.receipt()}
            if args.precision == "float16" or grammar or args.branch_strategy == "trie":
                row.update(reference_target_logprobs=reference_targets, branch_target_logprobs=branch_targets)
            if grammar:
                row.update(choice_comparisons=choice_comparisons, comparison_scope="within_decision",
                           cross_decision_comparison_diagnostic_only=comparison)
                del row["comparison"]
            if precision is not None:
                baseline = native_scores[source]
                row["native_precision_reference"] = baseline
                ranking_key = f"native_to_{args.precision}_ranking"
                row[ranking_key] = ranked_score_equivalence(
                    [value["score"] for value in baseline], reference_scores, tolerance=score_tolerance)
                row[f"native_to_{args.precision}_target_logprob_errors"] = native_precision_errors
                row["native_execution_equivalent"] = (
                    row[ranking_key]["accepted"] and max(native_precision_errors) <= tolerance)
                if grammar:
                    native_choices = partitioned_score_equivalence([value["score"] for value in baseline],
                        reference_scores, partitions[source], tolerance=score_tolerance)
                    row[ranking_key] = native_choices
                    row["native_execution_equivalent"] = (
                        all(value["comparison"]["accepted"] for value in native_choices)
                        and max(native_precision_errors) <= tolerance)
                row["prefix_dtype"] = str(full.dtype)
                row["logit_dtype"] = str(reference.dtype)
            rows.append(row)
            print(json.dumps({key: row[key] for key in ("source", "alternatives", "accepted",
                "target_logprob_errors", "full_seconds", "branch_seconds")} if grammar
                or args.branch_strategy == "trie" else row), flush=True)
            del branches
        current = get_active_cortex_spec(force_refresh=True)
        if (current is None or current.descriptor_sha256 != spec.descriptor_sha256
                or current.pointer_sha256 != spec.pointer_sha256
                or selected_checkpoint(args.training_directory) != (training, selected)
                or ("installed_arithmetic" in plan and installed_arithmetic_basis() != plan["installed_arithmetic"])
                or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
                       for name, sha in implementation.items())):
            raise ValueError("branch probe model, checkpoint, or implementation drifted")
        body = {"schema": "aura.native_prefix_branch_probe.v1", "plan_sha256": plan["plan_sha256"],
                "rows": rows, "accepted": all(row["accepted"] for row in rows),
                "elapsed_seconds": time.monotonic() - started, "serving_authority": False,
                "qualification_evidence": False}
        if precision is not None:
            body.update(schema="aura.native_prefix_branch_probe.v2", precision_receipt=precision_receipt,
                        precision_contract=precision,
                        native_execution_equivalent=all(row["native_execution_equivalent"] for row in rows),
                        production_optimization_authorized=False)
        if args.precision == "float16":
            body["schema"] = "aura.native_prefix_branch_probe.v3"
        if grammar or args.branch_strategy == "trie":
            body.update(schema="aura.native_prefix_branch_probe.v4", branch_strategy=args.branch_strategy,
                        ranking_scope="within_each_declared_inference_competition")
        _save_if_absent(args.directory / "report.json", {**body, "receipt_sha256": digest(body)})


if __name__ == "__main__":
    main()
