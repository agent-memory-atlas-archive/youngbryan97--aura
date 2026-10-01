"""Joint native/pointer source fitting with exact model and bounded shard custody."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path


class NativeCaptureBank(Mapping):
    def __init__(self, store, templates):
        self.store, self.templates = store, templates

    def __iter__(self):
        return iter(self.templates)

    def __len__(self):
        return len(self.templates)

    def __getitem__(self, identity):
        from core.learning.semantic_grounded_binding_engine import NativeGroundedCapture
        return NativeGroundedCapture(hidden=self.store[identity, 0, 0], **self.templates[identity])


def native_capture_template(item, evidence, depths):
    """Use public proposed operation/spans, not teacher argument registers."""
    instructions = item.ir.instructions
    anchors = (item.register_definition_spans or
               (*item.ir.input_spans, *(instruction.operation_span for instruction in instructions)))
    operations, mentions = {}, {}
    row = 0
    for instruction in instructions:
        for span in instruction.argument_spans:
            role = evidence.roles[row]
            operations[role.identity], mentions[role.identity] = instruction.operation_span, span
            row += 1
    if row != len(evidence.roles) or len(anchors) != len(evidence.context.referents):
        raise ValueError("native source spans differ from grounded role/candidate identities")
    return dict(source_id=evidence.source_id, context=evidence.context, roles=evidence.roles,
        depths=tuple(depths), operation_spans=operations, mention_spans=mentions,
        candidate_spans={record.key: span for record, span in zip(evidence.context.referents, anchors, strict=True)},
        adjacency=evidence.adjacency, relations=evidence.relations)


def fit_native_grounded_sources(training, calibration, source_items, directory, *, spec, rank=32, layers=8,
                               max_tokens=512, cache_bytes=512 * 1024 ** 2, seed=20260930,
                               relation_width=128, rounds=2, fit_options=None):
    """Load one authorized model; recompute actual suffix states in gradients.

    Prefixes are immutable complete source-only sequences, sharded on disk.
    This function runs source fitting, never held scoring or live activation.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load

    from core.governance_context import local_internal_governed_scope
    from core.learning.frozen_decoder_prefix import FrozenDecoderPrefix, NativeDecoderSuffix
    from core.learning.frozen_state_store import FrozenStateStore
    from core.learning.semantic_grounded_binding_engine import (
        fit_grounded_binding,
        implementation_receipt,
    )
    from core.learning.semantic_native_program import source_text_from_tokens
    from core.learning.semantic_relational_pointer import RelationalBindingPointer
    from core.runtime.file_write_gateway import get_file_write_gateway
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.semantic_native_adapters import (
        adapter_contract,
        install_native_adapters,
        native_adapter_parameter_estimate,
    )
    from tools.train_semantic_native_program import require_native_cortex_spec

    directory, fit_options = Path(directory), dict(fit_options or {})
    max_seconds = fit_options.get("max_seconds", 1800.)
    if (any(type(value) is not int or value < 1 for value in (rank, layers, max_tokens, cache_bytes))
            or not math.isfinite(max_seconds) or not 0 < max_seconds <= 14400):
        raise ValueError("native grounded fit needs bounded source geometry")
    examples = {item.evidence.source_id: item for item in (*training, *calibration)}
    items = {item.ir.source_text_sha256: item for item in source_items}
    if (not training or not calibration or len(examples) != len(training) + len(calibration)
            or set(items) != set(examples) or any(item.split != "train" for item in items.values())
            or directory.exists()
            or any(not item.ir.source_token_ids or len(item.ir.source_token_ids) > max_tokens for item in items.values())):
        raise ValueError("native grounded source custody, token bound or fresh fit directory differs")
    adaptation = adapter_contract(rank=rank, layers=layers, sites="native_topology_v1",
                                  scaling="alpha_over_sqrt_rank_v1", alpha=float(rank))
    plan = {"rank": rank, "suffix_layers": layers, "adapter_keys": adaptation["keys"],
        "adapter_contract": adaptation, "model_descriptor_sha256": spec.descriptor_sha256,
        "model_path": str(spec.model_path), "pointer_sha256": spec.pointer_sha256,
        "source_token_only": True, "implementation": implementation_receipt(), "seed": seed,
        "fit_ids": sorted(item.evidence.source_id for item in training),
        "calibration_ids": sorted(item.evidence.source_id for item in calibration),
        "max_tokens": max_tokens, "prefix_cache_bytes": cache_bytes,
        "relation_width": relation_width, "rounds": rounds,
        "fit_options": {**fit_options, "equivariance_pairs": [asdict(pair) for pair in fit_options.get("equivariance_pairs", ())]}}
    plan = json.loads(json.dumps(plan, allow_nan=False))
    plan_hash = hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    config = json.loads((spec.model_path / "config.json").read_text())
    projection = native_adapter_parameter_estimate(config, adaptation)
    if projection is None:
        raise ValueError("native grounded adapter topology has no measured parameter projection")
    custody = directory.parent / (directory.name + "-native-custody")
    if custody.exists():
        raise FileExistsError(custody)
    with local_internal_governed_scope("grounded_native_fit_plan", domain="file_write"):
        if not get_file_write_gateway().write_bytes_if_absent(custody / "plan.json",
            json.dumps({**plan, "plan_sha256": plan_hash}, indent=2).encode(),
            source="grounded_native_fit_plan", mode=0o400):
            raise FileExistsError(custody / "plan.json")
    started = time.monotonic()
    with (standalone_model_lane(owner_id=f"grounded-native:{directory.name}", model_path=str(spec.model_path),
            purpose="training", preemptible=False, require_exclusive=True, allow_owner_eviction=False,
            metadata={"tool": "fit_semantic_grounded_binding", "production_effect": False}),
          mlx_memory_envelope(fraction=.80) as envelope):
        if require_native_cortex_spec().descriptor_sha256 != spec.descriptor_sha256:
            raise ValueError("native grounded descriptor changed before model acquisition")
        model, tokenizer = load(str(spec.model_path))
        model.freeze()
        model.eval()
        prefix = FrozenDecoderPrefix(model, split_at=len(model.layers) - layers)
        suffix = NativeDecoderSuffix(model, split_at=len(model.layers) - layers)
        mx.random.seed(seed)
        install_native_adapters(model, plan)
        observed = sum(value.size for _, value in tree_flatten(suffix.trainable_parameters()))
        if observed != projection["trainable_parameters"] or any("lora_" not in key for key, _ in tree_flatten(model.trainable_parameters())):
            raise ValueError("native grounded adapter ownership differs from its projected sites")
        if mx.get_active_memory() + projection["five_float32_copies_bytes"] + cache_bytes > envelope.memory_bytes:
            raise MemoryError("native grounded fixed residency exceeds the host envelope")
        store = FrozenStateStore(custody / "prefixes", plan_sha256=plan_hash, max_resident_bytes=cache_bytes)
        templates = {}
        for identity, item in sorted(items.items()):
            if time.monotonic() - started >= max_seconds:
                raise TimeoutError("native grounded acquisition reached its declared bound")
            source_text_from_tokens(item, tokenizer)
            tokens = mx.array([item.ir.source_token_ids], dtype=mx.int32)
            hidden = prefix.capture(tokens)
            digest = hashlib.sha256(json.dumps(item.ir.source_token_ids).encode()).hexdigest()
            store.write_source(identity, {(identity, 0, 0): hidden}, sequence_digests={(identity, 0, 0): digest})
            templates[identity] = native_capture_template(item, examples[identity].evidence, tuple(range(layers)))
            print(json.dumps({"stage": "grounded_prefix_captured", "source": identity,
                              "source_tokens": len(item.ir.source_token_ids), "active_bytes": mx.get_active_memory()}), flush=True)
        remaining = max_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("native grounded acquisition exhausted its complete fit allowance")
        pointer = RelationalBindingPointer(hidden.shape[-1], depths=layers, relation_width=relation_width, rounds=rounds)
        engine, report = fit_grounded_binding(pointer, training, calibration, directory,
            native_suffix=suffix, native_captures=NativeCaptureBank(store, templates), native_contract=plan,
            **{**fit_options, "max_seconds": remaining})
        if require_native_cortex_spec().descriptor_sha256 != spec.descriptor_sha256 or implementation_receipt() != plan["implementation"]:
            raise ValueError("native grounded model or implementation changed during source fitting")
        receipt = {"schema": "aura.grounded_native_acquisition.v1", "plan_sha256": plan_hash,
            "fit_receipt_sha256": report["receipt_sha256"], "adapter_projection": projection,
            "observed_adapter_parameters": observed, "state_store": store.receipt(),
            "memory_envelope": envelope.to_receipt(), "peak_memory_bytes": mx.get_peak_memory(),
            "held_sources_scored": False, "serving_authority": False}
        with local_internal_governed_scope("grounded_native_fit_receipt", domain="file_write"):
            if not get_file_write_gateway().write_bytes_if_absent(custody / "completion.json",
                json.dumps(receipt, indent=2).encode(), source="grounded_native_fit_receipt", mode=0o400):
                raise FileExistsError(custody / "completion.json")
        # The offline tool must not hand a model to a caller after releasing
        # ownership. Only source-fit artifacts and receipts leave this scope.
        del engine, model, suffix, prefix, hidden, store
        mx.clear_cache()
    return None, report
