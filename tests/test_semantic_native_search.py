"""Native search explores bindings, preserves raw evidence, and reports limits."""

import math

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_grammar import (
    NativeGrammarUnreachableError,
    decode_native_grammar,
    minimum_graph_completion_steps,
)
from core.learning.semantic_native_search import _native_search_bound_truth, search_native_grammar


def test_search_limit_truth_is_registered_as_an_executable_invariant():
    assert _native_search_bound_truth() == ()


def test_one_step_search_matches_exhaustive_normalized_scores():
    calls = []
    def score(choices):
        calls.append(choices)
        return tuple(3. if choice.value == "neg" else 0. for choice in choices)
    search = search_native_grammar(("integer",), score, max_steps=1,
                                   max_nodes=1000, completions=100)
    assert search.halt_reason == "frontier_exhausted"
    assert not search.requested_top_k_proven
    assert search.frontier_nodes == 0
    assert search.candidates[0].result.program.instructions[0].op == "neg"
    assert len(calls) == search.scored_decisions
    assert len(set(calls)) == len(calls)
    assert math.isclose(sum(math.exp(row.log_probability) for row in search.candidates), 1.)
    assert [row.log_probability for row in search.candidates] == sorted(
        (row.log_probability for row in search.candidates), reverse=True)
    assert search.candidates[0].result.trace[0]["scores"] == tuple(
        3. if choice.value == "neg" else 0. for choice in calls[0])


@pytest.mark.parametrize("encoding", ["absolute_v1", "role_relative_v1"])
def test_top_k_graphs_share_the_existing_executor_and_proven_bound(encoding):
    def score(choices):
        return tuple(2. if choice.value in ("sub", 0, "input:0") else 0. for choice in choices)
    search = search_native_grammar(("integer", "integer"), score,
        max_steps=1, max_nodes=1000, completions=3, register_encoding=encoding)
    assert search.requested_top_k_proven and len(search.candidates) == 3
    assert search.halt_reason == "requested_completions"
    assert search.candidates[-1].log_probability >= search.frontier_log_probability_bound
    for candidate in search.candidates:
        assert type(candidate.result.program.run((9, 2))) is int
        assert candidate.result.bound_forced_completion


def test_node_exhaustion_is_not_an_infeasibility_or_semantic_proof():
    search = search_native_grammar(("integer",) * 4,
        lambda choices: (0.,) * len(choices), max_nodes=1)
    assert search.candidates == ()
    assert search.halt_reason == "node_bound" and search.frontier_nodes > 0
    assert not search.requested_top_k_proven
    assert search.disconnected_leaves == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "1"])
def test_unmeasured_or_invalid_scores_cannot_be_searched(bad):
    with pytest.raises(ValueError, match="finite"):
        search_native_grammar(("integer",), lambda choices: (bad,) * len(choices))


def test_search_does_not_drop_a_low_scored_alternative_binding():
    def score(choices):
        return tuple(20. if choice.value == "sub" else
                     2. if choice.value == 0 else 0. for choice in choices)
    search = search_native_grammar(("integer", "integer"), score,
                                   max_steps=1, max_nodes=1000, completions=4)
    assert search.requested_top_k_proven
    assert {row.result.program.instructions[0].args for row in search.candidates} == {
        (0, 0), (0, 1), (1, 0), (1, 1)}


def test_finite_inputs_with_unrepresentable_probability_differences_are_refused():
    def score(choices):
        return tuple(1e308 if index == 0 else -1e308 for index in range(len(choices)))
    with pytest.raises(ValueError, match="finite precision"):
        search_native_grammar(("integer",), score)


def test_disconnected_leaf_does_not_erase_a_connected_alternative():
    def score(choices):
        # Prefer two independent operations; the other binding joins them.
        operation = choices[0].value
        wanted = "neg" if isinstance(operation, str) and operation != "continue" else (
            "continue" if operation == "finish" else 0)
        return tuple(100. if choice.value == wanted else
                     90. if choice.value == 2 else -100. for choice in choices)
    search = search_native_grammar(("integer",) * 2, score,
                                   max_steps=2, max_nodes=1000, completions=2)
    assert search.disconnected_leaves > 0
    assert search.candidates
    assert any(row.result.program.instructions[-1].args == (2,) for row in search.candidates)


def test_completion_lower_bound_counts_independent_outputs_not_inputs():
    independent = Program(2, (Instruction("neg", (0,)), Instruction("neg", (1,)),
                              Instruction("neg", (0,))))
    joined = Program(2, (Instruction("neg", (0,)), Instruction("neg", (1,)),
                         Instruction("add", (2, 3))))
    assert minimum_graph_completion_steps(independent) == 2
    assert minimum_graph_completion_steps(joined) == 0


def test_native_decoder_prunes_only_after_a_partial_graph_becomes_unreachable():
    def independent(choices):
        return tuple(10. if choice.value in {"neg", "continue", 0} else 0. for choice in choices)

    with pytest.raises(NativeGrammarUnreachableError) as failure:
        decode_native_grammar(("integer", "integer"), independent, max_steps=4,
                              viability_pruning=True)
    assert failure.value.remaining_steps == 1
    assert failure.value.minimum_additional_steps == 2
    assert len(failure.value.program.instructions) == 3


def test_native_search_keeps_connected_graphs_with_viability_pruning():
    def prefer_independent(choices):
        return tuple(10. if choice.value in {"neg", "continue", 0} else 0. for choice in choices)

    search = search_native_grammar(("integer", "integer"), prefer_independent,
                                   max_steps=4, max_nodes=1000, completions=1)
    assert search.pruned_prefixes > 0
    assert search.candidates
    assert all(minimum_graph_completion_steps(row.result.program) == 0
               for row in search.candidates)


def test_viability_pruning_preserves_unbounded_top_graph_order():
    def score(choices):
        return tuple(2. if choice.value in {"neg", "continue", 0} else 0. for choice in choices)

    options = {"max_steps": 2, "max_nodes": 10000, "completions": 6}
    pruned = search_native_grammar(("integer", "integer"), score, **options)
    full = search_native_grammar(("integer", "integer"), score,
                                 viability_pruning=False, **options)
    assert pruned.requested_top_k_proven and full.requested_top_k_proven
    assert [(row.result.program, row.log_probability) for row in pruned.candidates] == [
        (row.result.program, row.log_probability) for row in full.candidates]


def test_native_scores_keep_their_likelihood_scale_without_local_renormalization():
    def score(choices):
        return tuple(-1. if choice.value == "neg" else -5. for choice in choices)
    search = search_native_grammar(("integer",), score, max_steps=1,
        max_nodes=1000, completions=1, score_mode="native_nonpositive")
    assert search.requested_top_k_proven
    assert search.candidates[0].result.program.instructions[0].op == "neg"
    assert search.candidates[0].log_probability == -6.


def test_positive_scores_cannot_supply_an_admissible_native_path_bound():
    with pytest.raises(ValueError, match="nonpositive"):
        search_native_grammar(("integer",), lambda choices: (1.,) * len(choices),
                              score_mode="native_nonpositive")
    with pytest.raises(ValueError, match="mode"):
        search_native_grammar(("integer",), lambda choices: (0.,) * len(choices), score_mode="auto")
