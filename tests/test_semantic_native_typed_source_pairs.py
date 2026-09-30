"""Typed source coverage must teach binding without using held-source supervision."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_counterfactual_corpus import augment_source_programs
from core.learning.semantic_native_typed_source_pairs import (
    _typed_contrast_coverage,
    native_typed_source_pair_plan,
    typed_source_pair_inventory,
)
from core.learning.semantic_program_corpus import build_semantic_program_fork_join_corpus
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


def test_distinct_reference_roles_survive_in_one_fit_source():
    def source(identity, program, tokens):
        return SimpleNamespace(ir=SimpleNamespace(source_text_sha256=identity,
            source_token_ids=tokens, to_program=lambda: program),
            public_inputs=(8, 3, 2, 1), contrast_id="same", split="train")

    first = Instruction("sub", (0, 1))
    second = Instruction("mul", (2, 3))
    rows = (
        source("base", Program(4, (first, second, Instruction("sub", (4, 5)))), (1, 2, 3)),
        source("early", Program(4, (Instruction("sub", (1, 0)), second,
                                  Instruction("sub", (4, 5)))), (1, 4, 3)),
        source("join", Program(4, (first, second, Instruction("sub", (5, 4)))), (1, 2, 5)),
    )
    pairs = native_typed_source_pair_plan(rows, ("base", "early", "join"),
                                          register_encoding="absolute_v1")
    reference = [row for row in pairs["base"] if row["kind"] == "reference"]
    assert [(row["decision_index"], row["partner"]) for row in reference] == [
        (1, "early"), (9, "join")]
    inventory = typed_source_pair_inventory(pairs, ("base", "early", "join"))
    assert inventory["sources_by_decision"]["9"] >= 1
    assert inventory["interactions_by_kind"]["reference"] > inventory["sources_by_kind"]["reference"]


def test_rendered_fork_join_role_flip_becomes_late_native_source_contrast():
    original = next(item for item in build_semantic_program_fork_join_corpus(
        seed=41, source_order_registers=True) if item.split == "train"
        and item.instructions[-1].instruction.op == "sub")
    rendered, receipt = augment_source_programs((original,), seed=12,
        lineage_version=2, mutation_policy="all_witnessed")
    unchanged = next(item for item in rendered if item.program == original.program)
    record = next(row for row in receipt["records"] if row.get("mutation_step") == 2
                  and row.get("mutation_kind") == "role")
    changed = next(item for item in rendered if item.example_id == record["example"])

    def training(item):
        identity = hashlib.sha256(item.source_text.encode()).hexdigest()
        return SimpleNamespace(ir=SimpleNamespace(source_text_sha256=identity,
            source_token_ids=tuple(item.source_text.encode()), to_program=lambda: item.program),
            public_inputs=item.inputs, contrast_id=item.contrast_id, split=item.split)

    examples = (training(unchanged), training(changed))
    fit_ids = tuple(item.ir.source_text_sha256 for item in examples)
    pairs = native_typed_source_pair_plan(examples, fit_ids, register_encoding="absolute_v1")
    assert all(any(row["kind"] == "reference" and row["decision_index"] == 9
                   for row in pairs[identity]) for identity in fit_ids)


@pytest.mark.parametrize("fit_ids", [(), ("a", "a"), ("a", "missing"), ("a", "held")])
def test_complete_unique_training_partition_is_required(fit_ids):
    with pytest.raises(ValueError, match="unique complete fit-only"):
        native_typed_source_pair_plan(corpus(), fit_ids, register_encoding="absolute_v1")


def test_duplicate_source_records_cannot_replace_one_another():
    with pytest.raises(ValueError, match="unique complete fit-only"):
        plan(corpus() + (corpus()[0],))


def test_coverage_inventory_counts_updates_and_keeps_absent_kinds_zero():
    pairs = {"a": [{"kind": "reference", "partner": "b", "decision_index": 1}]}
    assert typed_source_pair_inventory(pairs, ("a", "b", "a")) == {
        "paired_sources": 1, "paired_updates": 2,
        "sources_by_kind": {"operation": 0, "reference": 1, "termination": 0},
        "interactions_by_kind": {"operation": 0, "reference": 2, "termination": 0},
        "sources_by_decision": {"1": 1},
        "interactions_by_decision": {"1": 2},
        "held_labels_used": False}


def test_registered_coverage_invariant_measures_absence():
    assert _typed_contrast_coverage()["sources_by_kind"]["termination"] == 0


@pytest.mark.parametrize("defect", ["duplicate_decision", "new_kind", "empty", "unscheduled"])
def test_coverage_cannot_invent_or_duplicate_interactions(defect):
    pairs = {"a": [{"kind": "reference", "partner": "b", "decision_index": 1}]}
    if defect == "duplicate_decision":
        pairs["a"].append(deepcopy(pairs["a"][0]))
    elif defect == "new_kind":
        pairs["a"][0]["kind"] = "construction"
    elif defect == "empty":
        pairs["a"] = []
    with pytest.raises(ValueError, match="inventory"):
        typed_source_pair_inventory(pairs, ("b",) if defect == "unscheduled" else ("a", "b"))
