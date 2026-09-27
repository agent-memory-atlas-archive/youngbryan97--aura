"""Carry immutable frozen states into a new, fully disclosed optimizer run."""

from __future__ import annotations

import json
from pathlib import Path

from tools.evaluate_semantic_native_checkpoint import digest, verified_document

_REPLACED_IMPLEMENTATIONS = frozenset({
    "tools/train_semantic_native_program.py",
    "core/learning/frozen_state_store.py",
    "core/learning/semantic_native_source_control.py",
    "tools/semantic_native_prefix_reuse.py",
    "core/learning/semantic_native_source_pairs.py",
})

_PAIR_PLAN_FIELDS = frozenset({
    "grammar_source_pair_contract",
    "grammar_source_pair_fit_partners",
    "grammar_source_pair_updates",
})


def prefix_reuse_contract(origin: Path, new_plan: dict) -> dict:
    origin = origin.resolve()
    prior = verified_document(origin / "plan.json", "plan_sha256")
    supervision = verified_document(origin / "supervision.json")
    left = {key: value for key, value in prior.items()
            if key not in {"plan_sha256", "implementation"}}
    right = {key: value for key, value in json.loads(json.dumps(new_plan)).items()
             if key not in {"plan_sha256", "implementation", "reused_prefix_contract"}}
    paired = right.get("schema") == "aura.semantic_native_fit_plan.v4"
    if paired:
        from core.learning.semantic_native_source_control import source_control_mode_from_plan

        if source_control_mode_from_plan(new_plan) != "source_text":
            raise ValueError("paired frozen reuse needs intact source evidence")
        if set(right) - set(left) != _PAIR_PLAN_FIELDS:
            raise ValueError("paired frozen reuse changes undeclared plan fields")
        for name in _PAIR_PLAN_FIELDS:
            right.pop(name)
        right["schema"] = "aura.semantic_native_fit_plan.v3"
        right["objective"] = "grammar_choices"
    differences = sorted(key for key in left.keys() | right.keys()
                         if left.get(key) != right.get(key))
    if (differences or prior.get("schema") != "aura.semantic_native_fit_plan.v3"
            or prior.get("prefix_storage_contract", {}).get("mode") != "source_shards"
            or prior.get("execution_contract", {}).get("prefix_strategy") != "trie"
            or supervision.get("plan_sha256") != prior["plan_sha256"]):
        raise ValueError(f"prior frozen capture differs from fresh fitting protocol: {differences}")
    changes = {name for name in prior["implementation"] | new_plan["implementation"]
               if prior["implementation"].get(name) != new_plan["implementation"].get(name)}
    if (changes - _REPLACED_IMPLEMENTATIONS
            or ("core/learning/semantic_native_source_pairs.py" in changes and not paired)):
        raise ValueError("prior frozen capture implementation changed outside its reader")
    expected = {(row["source"], row["decision_index"], row["choice_index"]): digest(row["tokens"])
                for row in supervision["rows"]}
    if len(expected) != len(supervision["rows"]):
        raise ValueError("prior frozen supervision repeats alternatives")
    from core.learning.frozen_state_store import FrozenStateStore

    store = FrozenStateStore.open_existing(origin / "prefix-states",
        plan_sha256=prior["plan_sha256"],
        max_resident_bytes=prior["prefix_storage_contract"]["max_resident_bytes"],
        sequence_digests=expected)
    manifest_inventory = digest([(source, store._shards[source]["receipt_sha256"])
                                 for source in sorted(store._shards)])
    captures = source_capture_receipts(origin, sources=sorted(store._shards),
                                       plan_sha256=prior["plan_sha256"])
    return {"schema": "aura.native_frozen_prefix_reuse.v1",
            "source_directory": str(origin),
            "source_plan_sha256": prior["plan_sha256"],
            "source_supervision_receipt_sha256": supervision["receipt_sha256"],
            "source_supervision_rows_sha256": digest(supervision["rows"]),
            "manifest_inventory_sha256": manifest_inventory,
            "capture_inventory_sha256": digest(captures),
            "source_count": len(store._shards), "sequence_count": len(store),
            "changed_implementation_paths": sorted(changes),
            "optimizer_state_reused": False,
            "fresh_complete_schedule_required": True,
            "serving_authority": False}


def source_capture_receipts(origin: Path, *, sources, plan_sha256: str) -> list[dict]:
    from core.runtime.file_read_gateway import open_stable_readonly_binary

    directory = Path(origin) / "prefix-receipts"
    names = {f"{source}.json" for source in sources}
    if {path.name for path in directory.glob("*.json")} != names:
        raise ValueError("reused prefix capture inventory differs")
    receipts = []
    for source in sources:
        with open_stable_readonly_binary(directory / f"{source}.json",
                                         max_bytes=1024 * 1024) as (handle, _identity):
            row = json.load(handle)
        if row.get("plan_sha256") != plan_sha256 or row.get("source") != source:
            raise ValueError("reused prefix capture receipt differs")
        receipts.append({key: value for key, value in row.items() if key != "plan_sha256"})
    return receipts


def open_reused_prefix(contract: dict, plan: dict, supervision: dict):
    """Recheck the bound source and every sequence before fitting begins."""
    origin = Path(contract["source_directory"])
    if prefix_reuse_contract(origin, {key: value for key, value in plan.items()
                                      if key != "reused_prefix_contract"}) != contract:
        raise ValueError("frozen prefix reuse contract changed")
    if digest(supervision["rows"]) != contract["source_supervision_rows_sha256"]:
        raise ValueError("fresh supervision differs from frozen prefix source")
    from core.learning.frozen_state_store import FrozenStateStore

    digests = {(row["source"], row["decision_index"], row["choice_index"]): digest(row["tokens"])
               for row in supervision["rows"]}
    return FrozenStateStore.open_existing(origin / "prefix-states",
        plan_sha256=contract["source_plan_sha256"],
        max_resident_bytes=plan["prefix_storage_contract"]["max_resident_bytes"],
        sequence_digests=digests)
