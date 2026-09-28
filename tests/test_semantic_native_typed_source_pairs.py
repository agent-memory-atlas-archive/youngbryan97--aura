"""Typed source coverage must teach binding without using held-source supervision."""

import hashlib
import json
from copy import deepcopy

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_typed_source_pairs import (
    _typed_contrast_coverage,
    native_typed_source_pair_plan,
    typed_source_pair_inventory,
)
from tests.test_semantic_native_source_pairs import example


def corpus():
    return (
        example("a", "same", Program(2, (Instruction("sub", (0, 1)),)), tokens=(1, 2, 3)),
        example("op", "same", Program(2, (Instruction("add", (0, 1)),)), tokens=(1, 8, 3)),
        example("ref", "elsewhere", Program(2, (Instruction("sub", (1, 0)),)), tokens=(9, 2, 3)),
        example("end", "long", Program(2, (Instruction("sub", (0, 1)),
                    Instruction("mul", (2, 1)))), tokens=(1, 2, 3, 5)),
        example("held", "same", Program(2, (Instruction("sub", (1, 0)),)),
                    tokens=(1, 2, 3), split="validation"),
    )


def plan(rows=None):
    rows = corpus() if rows is None else rows
    return native_typed_source_pair_plan(rows, ("a", "op", "ref", "end"),
                                         register_encoding="absolute_v1")


def test_source_can_receive_operations_references_and_termination_from_different_peers():
    pairs = plan()
    assert [(row["kind"], row["partner"]) for row in pairs["a"]] == [
        ("operation", "op"), ("reference", "ref"), ("termination", "end")]
    assert "held" not in pairs
    assert all(row["partner"] != "held" for rows in pairs.values() for row in rows)
    for rows in pairs.values():
        for row in rows:
            assert row["own_index"] != row["partner_index"]
            assert row["witness_sha256"] == hashlib.sha256(json.dumps(row["witness"],
                sort_keys=True, allow_nan=False).encode()).hexdigest()


def test_peer_and_input_iteration_order_do_not_choose_a_different_plan():
    assert plan(tuple(reversed(corpus()))) == plan()


def test_equivalent_reference_permutation_is_not_a_negative_example():
    rows = (example("a", "one", Program(2, (Instruction("add", (0, 1)),)), tokens=(1, 2)),
            example("b", "two", Program(2, (Instruction("add", (1, 0)),)), tokens=(2, 1)))
    assert native_typed_source_pair_plan(rows, ("a", "b"), register_encoding="absolute_v1") == {}


def test_unknown_comparison_never_becomes_a_witness(monkeypatch):
    monkeypatch.setattr("core.learning.semantic_native_typed_source_pairs.compare_program_meanings",
                        lambda *args: {"status": "unknown", "witness": None})
    assert plan() == {}


def test_first_divergence_cannot_cross_an_already_different_prefix():
    pairs = plan()
    # 'op' differs before 'ref'; it cannot supply a reference negative to 'a'.
    assert next(row for row in pairs["a"] if row["kind"] == "reference")["partner"] == "ref"
    assert all(row["kind"] != "reference" for row in pairs["op"])


@pytest.mark.parametrize("fit_ids", [(), ("a", "a"), ("a", "missing"), ("a", "held")])
def test_complete_unique_training_partition_is_required(fit_ids):
    with pytest.raises(ValueError, match="unique complete fit-only"):
        native_typed_source_pair_plan(corpus(), fit_ids, register_encoding="absolute_v1")


def test_duplicate_source_records_cannot_replace_one_another():
    with pytest.raises(ValueError, match="unique complete fit-only"):
        plan(corpus() + (corpus()[0],))


def test_coverage_inventory_counts_updates_and_keeps_absent_kinds_zero():
    pairs = {"a": [{"kind": "reference", "partner": "b"}]}
    assert typed_source_pair_inventory(pairs, ("a", "b", "a")) == {
        "paired_sources": 1, "paired_updates": 2,
        "sources_by_kind": {"operation": 0, "reference": 1, "termination": 0},
        "interactions_by_kind": {"operation": 0, "reference": 2, "termination": 0},
        "held_labels_used": False}


def test_registered_coverage_invariant_measures_absence():
    assert _typed_contrast_coverage()["sources_by_kind"]["termination"] == 0


@pytest.mark.parametrize("defect", ["duplicate_kind", "new_kind", "empty", "unscheduled"])
def test_coverage_cannot_invent_or_duplicate_interactions(defect):
    pairs = {"a": [{"kind": "reference", "partner": "b"}]}
    if defect == "duplicate_kind":
        pairs["a"].append(deepcopy(pairs["a"][0]))
    elif defect == "new_kind":
        pairs["a"][0]["kind"] = "construction"
    elif defect == "empty":
        pairs["a"] = []
    with pytest.raises(ValueError, match="inventory"):
        typed_source_pair_inventory(pairs, ("b",) if defect == "unscheduled" else ("a", "b"))
