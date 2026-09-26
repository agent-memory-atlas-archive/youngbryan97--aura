#!/usr/bin/env python3
"""Measure frozen source reuse against single-row full-sequence inference."""

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
    for (source, program), sequence in supervision_sequences(supervision).items():
        groups.setdefault(source, {})[program] = sequence
    eligible = [(source, tuple(group[key] for key in sorted(group)))
                for source, group in sorted(groups.items()) if len(group) >= 2]
    if len(eligible) < count:
        raise ValueError("branch probe lacks complete source-bound alternatives")
    return tuple(eligible[:count])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=1800.)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.sources <= 16 or not 0 < args.max_seconds <= 3600:
        parser.error("branch probe needs bounded sources and runtime")
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
            "ranking_must_be_identical": True, "serving_authority": False,
            "qualification_evidence": False}
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
        for source, sequences in groups:
            begin = time.monotonic()
            branches = FrozenPrefixBranches(model, split_at=split,
                anchor_tokens=native_source_anchor(sequences), max_tokens=training["max_sequence_tokens"])
            branch_seconds, full_seconds = time.monotonic() - begin, 0.
            reference_scores, branch_scores, errors, score_tolerance = [], [], [], 0.
            for sequence in sequences:
                if time.monotonic() - started > args.max_seconds:
                    raise TimeoutError("branch probe reached its finite runtime bound")
                positions = tuple(index - 1 for index in native_prediction_positions(sequence,
                    scope=training["loss_scope"]))
                targets = mx.array([sequence.tokens[index + 1] for index in positions], dtype=mx.int32)
                begin = time.monotonic()
                full = prefix.capture(mx.array([sequence.tokens[:-1]], dtype=mx.int32))
                reference = suffix(full, logit_positions=positions)
                mx.eval(reference)
                full_seconds += time.monotonic() - begin
                begin = time.monotonic()
                cached = branches.capture(tuple(sequence.tokens[:-1]))
                alternate = suffix(cached, logit_positions=positions)
                mx.eval(alternate)
                branch_seconds += time.monotonic() - begin
                def target_logprob(logits, *, target_ids=targets):
                    values = logits[0].astype(mx.float32)
                    return mx.take_along_axis(values, target_ids[:, None], axis=1)[:, 0] - mx.logsumexp(values, axis=-1)
                left, right = target_logprob(reference), target_logprob(alternate)
                tolerance = 2 * max(mx.finfo(reference.dtype).eps, mx.finfo(alternate.dtype).eps)
                errors.append(float(mx.max(mx.abs(left - right)).item()))
                score_tolerance = max(score_tolerance, tolerance * len(positions))
                reference_scores.append(float(mx.sum(left).item()))
                branch_scores.append(float(mx.sum(right).item()))
            comparison = ranked_score_equivalence(reference_scores, branch_scores, tolerance=score_tolerance)
            row = {"source": source, "alternatives": len(sequences), "reference_scores": reference_scores,
                   "branch_scores": branch_scores, "target_logprob_errors": errors,
                   "target_tolerance": tolerance, "comparison": comparison,
                   "accepted": comparison["accepted"] and max(errors) <= tolerance,
                   "full_seconds": full_seconds, "branch_seconds": branch_seconds,
                   "cache": branches.receipt()}
            rows.append(row)
            print(json.dumps(row), flush=True)
            del branches
        current = get_active_cortex_spec(force_refresh=True)
        if (current is None or current.descriptor_sha256 != spec.descriptor_sha256
                or current.pointer_sha256 != spec.pointer_sha256
                or selected_checkpoint(args.training_directory) != (training, selected)
                or any(hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha
                       for name, sha in implementation.items())):
            raise ValueError("branch probe model, checkpoint, or implementation drifted")
        body = {"schema": "aura.native_prefix_branch_probe.v1", "plan_sha256": plan["plan_sha256"],
                "rows": rows, "accepted": all(row["accepted"] for row in rows),
                "elapsed_seconds": time.monotonic() - started, "serving_authority": False,
                "qualification_evidence": False}
        _save_if_absent(args.directory / "report.json", {**body, "receipt_sha256": digest(body)})


if __name__ == "__main__":
    main()
