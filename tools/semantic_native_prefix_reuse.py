"""Carry immutable frozen states into a new, fully disclosed optimizer run."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from tools.evaluate_semantic_native_checkpoint import digest, verified_document

_REPLACED_IMPLEMENTATIONS = frozenset({
    "tools/train_semantic_native_program.py",
    "core/learning/frozen_state_store.py",
    "core/learning/semantic_native_source_control.py",
    "tools/semantic_native_prefix_reuse.py",
    "core/learning/semantic_native_source_pairs.py",
    "core/learning/semantic_native_typed_source_pairs.py",
    "core/learning/semantic_native_path_objective.py",
    "core/learning/semantic_native_path_calibration.py",
    "core/learning/semantic_native_path_selection.py",
})

_PAIR_PLAN_FIELDS = frozenset({
    "grammar_source_pair_contract",
    "grammar_source_pair_fit_partners",
    "grammar_source_pair_updates",
})

_PATH_IMPLEMENTATIONS = frozenset({
    "core/learning/semantic_native_path_objective.py",
    "core/learning/semantic_native_path_calibration.py",
    "core/learning/semantic_native_path_selection.py",
})


def annotation_erased_numeric_ast(source: str) -> str:
    """Compare executable structure under postponed, unreflected annotations.

    This covers the numeric prefix call path, not annotation introspection.
    Class field annotations remain intact: dataclasses can depend on them.
    """
    tree = ast.parse(source)
    if not any(isinstance(node, ast.ImportFrom) and node.module == "__future__"
               and any(alias.name == "annotations" for alias in node.names) for node in tree.body):
        raise ValueError("numeric annotation reuse needs postponed annotations")
    if any(isinstance(node, ast.Name) and node.id == "TYPE_CHECKING"
           and isinstance(node.ctx, ast.Store) for node in ast.walk(tree)):
        raise ValueError("annotation-only guard is rebound at runtime")

    class Erase(ast.NodeTransformer):
        def visit_arg(self, node):
            node.annotation = None
            return node

        def visit_FunctionDef(self, node):
            self.generic_visit(node)
            node.returns = None
            return node

        def visit_AsyncFunctionDef(self, node):
            return self.visit_FunctionDef(node)

        def visit_If(self, node):
            if isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
                if node.orelse or any(not isinstance(item, (ast.Import, ast.ImportFrom)) for item in node.body):
                    raise ValueError("annotation-only guard contains executable behavior")
                return None
            return self.generic_visit(node)

    tree = Erase().visit(tree)
    loads = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in {"typing", "collections.abc"}:
            node.names = [alias for alias in node.names if (alias.asname or alias.name) in loads]
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or node.names]
    return ast.dump(tree, include_attributes=False)


def annotation_reuse_paths(prior, new_plan, changes, *, root=None):
    binding = new_plan.get("annotation_only_prefix_sources")
    if binding is None:
        return []
    archive = verified_document(Path(binding["path"]))
    if (archive.get("schema") != "aura.native_prefix_annotation_sources.v1"
            or archive["receipt_sha256"] != binding["receipt_sha256"]
            or archive.get("source_plan_sha256") != prior["plan_sha256"]):
        raise ValueError("annotation reuse source archive differs")
    sources = archive["sources"]
    required = changes - _REPLACED_IMPLEMENTATIONS
    if set(sources) != required:
        raise ValueError("annotation reuse source inventory differs")
    root = Path(__file__).resolve().parent.parent if root is None else root
    for name, original in sources.items():
        current = (root / name).read_text()
        if (hashlib.sha256(original.encode()).hexdigest() != prior["implementation"].get(name)
                or hashlib.sha256(current.encode()).hexdigest() != new_plan["implementation"].get(name)
                or annotation_erased_numeric_ast(original) != annotation_erased_numeric_ast(current)):
            raise ValueError(f"frozen numeric implementation differs beyond annotations: {name}")
    return sorted(required)


def prefix_reuse_contract(origin: Path, new_plan: dict) -> dict:
    origin = origin.resolve()
    prior = verified_document(origin / "plan.json", "plan_sha256")
    supervision = verified_document(origin / "supervision.json")
    left = {key: value for key, value in prior.items()
            if key not in {"plan_sha256", "implementation"}}
    right = {key: value for key, value in json.loads(json.dumps(new_plan)).items()
             if key not in {"plan_sha256", "implementation", "reused_prefix_contract"}}
    typed_mode = right.get("schema") == "aura.semantic_native_fit_plan.v6"
    path_mode = right.get("schema") in {"aura.semantic_native_fit_plan.v5", "aura.semantic_native_fit_plan.v6"}
    paired = right.get("schema") == "aura.semantic_native_fit_plan.v4" or (
        path_mode and right.get("objective") == "grammar_source_pairs")
    if path_mode:
        from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
        from core.learning.semantic_native_source_control import source_control_mode_from_plan

        source_control_mode_from_plan(new_plan)
        if typed_mode:
            right.pop("grammar_source_pair_inventory")
        for name in ("grammar_path_objective_contract", "path_checkpoint_selection_contract"):
            right.pop(name)
        right.pop("annotation_only_prefix_sources", None)
        right["grammar_choice_contract"] = dict(GRAMMAR_CHOICE_CONTRACT)
        right["selection"] = "minimum_source_calibration_conditional_grammar_choice_loss"
        right["schema"] = "aura.semantic_native_fit_plan.v4" if paired else "aura.semantic_native_fit_plan.v3"
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
    annotations = annotation_reuse_paths(prior, new_plan, changes) if path_mode else []
    if (changes - _REPLACED_IMPLEMENTATIONS - set(annotations)
            or ("core/learning/semantic_native_source_pairs.py" in changes and not paired)
            or ("core/learning/semantic_native_typed_source_pairs.py" in changes and not typed_mode)
            or (changes & _PATH_IMPLEMENTATIONS and not path_mode)):
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
    result = {"schema": "aura.native_frozen_prefix_reuse.v1",
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
    if "annotation_only_prefix_sources" in new_plan:
        result["annotation_only_implementation_paths"] = annotations
        result["annotation_introspection_equivalence_claimed"] = False
    return result


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
