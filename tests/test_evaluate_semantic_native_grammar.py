"""Complete graph selection sees programs and scores, never their targets."""

from types import SimpleNamespace

import pytest

from tools.evaluate_semantic_native_grammar import (
    grammar_examples,
    grammar_pair_totals,
    select_search_proposal,
    source_input_types,
    validated_source_pair_map,
)
from tools.verify_semantic_native_grammar import verified_source_evidence


def search(programs):
    return SimpleNamespace(candidates=tuple(SimpleNamespace(
        result=SimpleNamespace(program=program)) for program in programs))


def test_every_program_is_scored_before_choosing_a_later_alternative():
    calls = []
    def score(program):
        calls.append(program)
        return {"a": -3., "b": -1., "c": -2.}[program]
    assert select_search_proposal(search(("a", "b", "c")), score) == (1, (-3., -1., -2.))
    assert calls == ["a", "b", "c"]


def test_empty_search_does_not_invent_a_candidate():
    assert select_search_proposal(search(()), lambda _: pytest.fail("no graph exists")) == (None, ())


def test_source_control_mode_must_match_the_plan_report_and_each_row():
    assert verified_source_evidence({}, {}, ({},)) == "source_text"
    plan = {"source_evidence": "source_token_erasure"}
    report = {"source_evidence": "source_token_erasure"}
    rows = ({"source_evidence": "source_token_erasure"},)
    assert verified_source_evidence(plan, report, rows) == "source_token_erasure"
    with pytest.raises(ValueError, match="source-evidence mode differs"):
        verified_source_evidence(plan, report, ({"source_evidence": "source_text"},))
    with pytest.raises(ValueError, match="source-evidence mode differs"):
        verified_source_evidence({"source_evidence": "unknown"}, report, rows)


@pytest.mark.parametrize("invalid", [True, None, float("nan"), float("inf")])
def test_unmeasured_graphs_cannot_be_skipped_to_claim_a_winner(invalid):
    with pytest.raises(ValueError, match="finite"):
        select_search_proposal(search(("a", "b")), lambda program: 1. if program == "a" else invalid)


def test_public_source_supplies_native_types_and_values():
    public, types = source_input_types("Use [7, -2] with 3.")
    assert public.values == ((7, -2), 3)
    assert types == ("integer_sequence", "integer")
    assert public.receipt()["source_text_sha256"] == public.source_text_sha256
    with pytest.raises(ValueError, match="no supported public values"):
        source_input_types("There are no literals in this request.")


def test_intervention_population_is_pair_complete_and_source_distinct():
    examples = grammar_examples(dataset="operation_intervention", seed=2718283, count=6)
    assert len(examples) == 6
    assert {example.topology_id for example in examples} == {
        "scalar_linear_three", "lookup_linear_three", "count_linear_three",
    }
    assert all(examples[index].construction_id == examples[index + 1].construction_id
               and examples[index].source_text != examples[index + 1].source_text
               for index in range(0, len(examples), 2))
    with pytest.raises(ValueError, match="complete source pairs"):
        grammar_examples(dataset="operation_intervention", seed=2718283, count=5)
    with pytest.raises(ValueError, match="exceeds"):
        grammar_examples(dataset="operation_intervention", seed=2718283, count=50)


def test_source_pair_swap_requires_a_meaning_change_at_fixed_public_inputs():
    from core.learning.procedure_induction import Instruction, Program
    from hashlib import sha256

    add = Program(2, (Instruction("add", (0, 1)),))
    sub = Program(2, (Instruction("sub", (0, 1)),))
    left = SimpleNamespace(source_text="Add 8 and 3.", program=add)
    right = SimpleNamespace(source_text="Subtract 8 from 3.", program=sub)
    left_id = sha256(left.source_text.encode()).hexdigest()
    right_id = sha256(right.source_text.encode()).hexdigest()
    assert validated_source_pair_map((left, right), dataset="operation_intervention",
                                     require_contrast=True) == {left_id: right_id, right_id: left_id}
    with pytest.raises(ValueError, match="matched inputs"):
        validated_source_pair_map((left, SimpleNamespace(source_text="Subtract 9 from 3.",
                                                        program=sub)),
                                  dataset="operation_intervention", require_contrast=True)
    with pytest.raises(ValueError, match="distinct programs"):
        validated_source_pair_map((left, SimpleNamespace(source_text=right.source_text,
                                                        program=add)),
                                  dataset="operation_intervention", require_contrast=True)
    with pytest.raises(ValueError, match="intervention dataset"):
        validated_source_pair_map((left, right), dataset="natural_request",
                                  require_contrast=True)


@pytest.mark.parametrize("dataset", ["definition_intervention", "equation_intervention"])
def test_paraphrase_populations_retain_three_topologies_and_pair_order(dataset):
    examples = grammar_examples(dataset=dataset, seed=2718283, count=6)
    assert len(examples) == 6
    assert {example.topology_id for example in examples} == {
        "scalar_linear_three", "lookup_linear_three", "count_linear_three",
    }
    assert all(examples[index].construction_id == examples[index + 1].construction_id
               and examples[index].source_text != examples[index + 1].source_text
               for index in range(0, 6, 2))
    with pytest.raises(ValueError, match="complete source pairs"):
        grammar_examples(dataset=dataset, seed=2718283, count=5)


def test_intervention_pair_totals_require_both_answers_and_a_source_response():
    before = {"program_equivalent": True, "answer_correct": True,
              "decode_status": "completed", "program": {"instructions": [
                  ["add", [0, 1]], ["sub", [4, 2]]]}}
    after = {**before, "program": {"instructions": [
        ["add", [0, 1]], ["add", [4, 2]]]}}
    assert grammar_pair_totals((before, after), dataset="operation_intervention") == {
        "pair_count": 1, "pair_exact": 1, "source_responsive": 1,
    }
    wrong = {**after, "answer_correct": False,
             "program": {"instructions": [["add", [0, 1]], ["add", [4, 3]]]}}
    assert grammar_pair_totals((before, wrong), dataset="operation_intervention") == {
        "pair_count": 1, "pair_exact": 0, "source_responsive": 0,
    }
    with pytest.raises(ValueError, match="split a source pair"):
        grammar_pair_totals((before,), dataset="operation_intervention")
