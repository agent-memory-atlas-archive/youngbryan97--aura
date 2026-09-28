"""Numeric reuse tolerates signatures, never masks, arithmetic, or class fields."""

import hashlib
import json

import pytest

from tools.evaluate_semantic_native_checkpoint import digest
from tools.semantic_native_prefix_reuse import annotation_erased_numeric_ast, annotation_reuse_paths

OLD = "from __future__ import annotations\ndef step(value, *, scale=1):\n return value * scale\n"
NEW = ("from __future__ import annotations\nfrom typing import TYPE_CHECKING, Any\n"
       "if TYPE_CHECKING:\n from imaginary.module import Numeric\n"
       "def step(value: Numeric, *, scale: int=1) -> Any:\n return value * scale\n")


def test_postponed_signature_and_type_only_imports_leave_numeric_ast_identical():
    assert annotation_erased_numeric_ast(OLD) == annotation_erased_numeric_ast(NEW)


@pytest.mark.parametrize("change", ["return value + scale", "return scale * value", "scale=2"])
def test_operation_order_or_default_changes_are_not_annotation_equivalence(change):
    current = NEW.replace("return value * scale", change) if change.startswith("return") else NEW.replace("scale: int=1", change)
    assert annotation_erased_numeric_ast(OLD) != annotation_erased_numeric_ast(current)


def test_dataclass_fields_are_never_erased_for_reuse():
    original = "from __future__ import annotations\nclass State:\n value: int\n"
    current = original.replace("value: int", "value: str")
    assert annotation_erased_numeric_ast(original) != annotation_erased_numeric_ast(current)


@pytest.mark.parametrize("source", ["def step(value: int): return value",
                                   NEW.replace("from imaginary.module import Numeric", "value = 9"),
                                   NEW + "TYPE_CHECKING = True\n"])
def test_eager_or_executable_annotation_changes_are_refused(source):
    with pytest.raises(ValueError):
        annotation_erased_numeric_ast(source)


@pytest.mark.parametrize("defect", [None, "original", "current", "binding", "inventory"])
def test_archive_binds_the_old_source_bytes_and_the_new_numeric_implementation(tmp_path, defect):
    name = "prefix.py"
    (tmp_path / name).write_text(NEW)
    prior = {"plan_sha256": "old", "implementation": {name: hashlib.sha256(OLD.encode()).hexdigest()}}
    archive = {"schema": "aura.native_prefix_annotation_sources.v1", "source_plan_sha256": "old",
               "sources": {name: OLD}}
    if defect == "original":
        archive["sources"][name] = OLD.replace("*", "+")
    if defect == "inventory":
        archive["sources"] = {}
    path = tmp_path / "archive.json"
    path.write_text(json.dumps({**archive, "receipt_sha256": digest(archive)}))
    plan = {"implementation": {name: hashlib.sha256(NEW.encode()).hexdigest()},
            "annotation_only_prefix_sources": {"path": str(path), "receipt_sha256": digest(archive)}}
    if defect == "current":
        (tmp_path / name).write_text(NEW.replace("*", "+"))
    if defect == "binding":
        plan["annotation_only_prefix_sources"]["receipt_sha256"] = "different"
    if defect is None:
        assert annotation_reuse_paths(prior, plan, {name}, root=tmp_path) == [name]
    else:
        with pytest.raises(ValueError):
            annotation_reuse_paths(prior, plan, {name}, root=tmp_path)
