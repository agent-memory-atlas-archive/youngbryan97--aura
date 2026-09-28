"""Role and dependency controls isolate witnessed graph changes, not operations."""

import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent
from core.learning.semantic_public_inputs import semantic_public_character_inputs
from tools.evaluate_semantic_native_grammar import grammar_examples, grammar_pair_totals
from tools.semantic_native_graph_interventions import (
    build_native_graph_interventions,
    graph_intervention_candidate,
)
from tools.verify_semantic_native_grammar import (
    verified_dataset,
    verified_examples,
    verified_pair_totals,
    verified_prefix_execution,
    verified_trie_anchor,
)


@pytest.mark.parametrize("kind", ["role", "dependency"])
@pytest.mark.parametrize("seed", [1618033, 2718283, 0])
def test_generated_pairs_change_one_binding_and_preserve_every_public_literal(kind, seed):
    pairs = build_native_graph_interventions(seed=seed, kind=kind)
    assert len(pairs) == 24
    assert sorted(Counter(pair.original.topology_id for pair in pairs).values()) == [8, 8, 8]
    assert len({row.source_text for pair in pairs for row in (pair.original, pair.changed)}) == 48
    for pair in pairs:
        left, right = pair.original, pair.changed
        assert left.inputs == right.inputs
        assert left.program.depth == right.program.depth == 3
        assert [step.op for step in left.program.instructions] == [step.op for step in right.program.instructions]
        assert [i for i, (a, b) in enumerate(zip(left.program.instructions, right.program.instructions, strict=True))
                if a != b] == [pair.changed_instruction]
        before, after = (row.program.instructions[pair.changed_instruction].args for row in (left, right))
        assert (before == after[::-1] if kind == "role" else sum(a != b for a, b in zip(before, after, strict=True)) == 1)
        assert not semantic_programs_structurally_equivalent(left.program, right.program)
        assert left.program.run(left.inputs) != right.program.run(right.inputs)
        for row in (left, right):
            assert semantic_public_character_inputs(row.source_text).values == row.inputs
            assert tuple(row.source_text[span.start:span.end] for span in row.input_spans) == tuple(
                left.source_text[span.start:span.end] for span in left.input_spans)


@pytest.mark.parametrize("kind", ["role", "dependency"])
def test_independent_verifier_rebuilds_population_and_graph_responsiveness(kind):
    dataset = kind + "_intervention"
    examples = grammar_examples(dataset=dataset, seed=1618033, count=6)
    plan = {"schema": "aura.semantic_native_grammar_plan.v5", "dataset": dataset,
            "seed": 1618033, "sources": [None] * 6}
    report = {"dataset": dataset, "seed": 1618033}
    assert verified_dataset(plan, report) == (dataset, 1618033)
    assert verified_examples(plan, dataset=dataset, seed=1618033) == examples
    rows = tuple({"program": example.program.to_dict(), "decode_status": "completed",
                  "program_equivalent": True, "answer_correct": True} for example in examples)
    expected = {"pair_count": 3, "pair_exact": 3, "source_responsive": 3}
    assert grammar_pair_totals(rows, dataset=dataset) == expected
    assert verified_pair_totals(rows, dataset=dataset) == expected
    repeated = tuple(row if i % 2 == 0 else rows[i - 1] for i, row in enumerate(rows))
    assert grammar_pair_totals(repeated, dataset=dataset)["source_responsive"] == 0
    assert verified_pair_totals(repeated, dataset=dataset)["source_responsive"] == 0
    with pytest.raises(ValueError, match="complete source pairs"):
        grammar_examples(dataset=dataset, seed=1618033, count=5)
    with pytest.raises(ValueError, match="dataset or seed"):
        verified_dataset({**plan, "schema": "aura.semantic_native_grammar_plan.v4"}, report)


def test_symmetric_binding_cannot_be_reported_as_a_changed_role():
    assert graph_intervention_candidate(Program(2, (Instruction("add", (0, 1)),)),
                                        (3, 7), kind="role") is None
    assert graph_intervention_candidate(Program(2, (Instruction("sub", (0, 1)),)),
                                        (3, 3), kind="role") is None
    with pytest.raises(ValueError, match="unsupported"):
        graph_intervention_candidate(Program(2, (Instruction("sub", (0, 1)),)), (3, 7), kind="unknown")


def test_trie_verifier_requires_strategy_and_complete_choice_coverage():
    plan = {"schema": "aura.semantic_native_grammar_plan.v7", "prefix_strategy": "trie"}
    report = {"prefix_strategy": "trie"}
    row = {"score_input_receipts": [[{}, {}], [{}]],
           "prefix_execution": {"schema": "aura.frozen_prefix_branches.v2",
                                "suffix_computation_unchanged": True,
                                "anchor_tokens": 4, "trie_calls": 2, "branches": 3}}
    verified_prefix_execution(plan, report)
    verified_prefix_execution(plan, report, row)
    for field, value in (("trie_calls", 1), ("branches", 2), ("anchor_tokens", 0),
                         ("suffix_computation_unchanged", False), ("schema", "v1")):
        with pytest.raises(ValueError, match="every choice"):
            verified_prefix_execution(plan, report, {
                **row, "prefix_execution": {**row["prefix_execution"], field: value}})
    with pytest.raises(ValueError, match="strategy differs"):
        verified_prefix_execution(plan, {"prefix_strategy": "full"})
    with pytest.raises(ValueError, match="historical"):
        verified_prefix_execution({"schema": "aura.semantic_native_grammar_plan.v5"}, report)


def test_trie_verifier_reconstructs_one_exact_source_anchor():
    sequences = (SimpleNamespace(tokens=(1, 2, 3, 4), continuation_start=3),
                 SimpleNamespace(tokens=(1, 2, 3, 5), continuation_start=3))
    sha = hashlib.sha256(json.dumps((1, 2, 3), separators=(",", ":"))
                         .encode("ascii")).hexdigest()
    receipt = {"anchor_tokens": 3, "anchor_token_sha256": sha}
    assert verified_trie_anchor(sequences, receipt) == sha
    assert verified_trie_anchor(sequences, receipt, sha) == sha
    for changed in ({"anchor_tokens": 2}, {"anchor_token_sha256": "wrong"}):
        with pytest.raises(ValueError, match="source anchor differs"):
            verified_trie_anchor(sequences, {**receipt, **changed})
    with pytest.raises(ValueError, match="source anchor differs"):
        verified_trie_anchor(sequences, receipt, "different")
