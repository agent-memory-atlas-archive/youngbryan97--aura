"""Smaller budget replay cannot invent scores or alter visited proposals."""

from types import SimpleNamespace

import pytest

from core.learning.semantic_native_search import search_native_grammar
from tools.evaluate_semantic_native_grammar import source_input_types
from tools.replay_semantic_native_budget_frontier import replay_budget


def test_budget_frontier_replays_only_a_saved_decision_prefix():
    source = "Add 2 and 3."
    transcript = []

    def score(choices):
        values = (0.,) * len(choices)
        transcript.append({"choices": [choice.value for choice in choices],
                           "scores": list(values)})
        return values

    plan = {"max_steps": 1, "search_nodes": 256, "search_completions": 4,
            "register_encoding": "role_relative_v1",
            "search_score_mode": "native_nonpositive"}
    full = search_native_grammar(source_input_types(source)[1], score,
        max_steps=plan["max_steps"], max_nodes=plan["search_nodes"],
        completions=plan["search_completions"],
        register_encoding=plan["register_encoding"],
        score_mode=plan["search_score_mode"])
    target = full.candidates[0].result.program
    row = {"search": {"score_transcript": transcript,
                      "proposals": [{"program": candidate.result.program.to_dict()}
                                    for candidate in full.candidates],
                      "complete_graph_scores": [0., -1., -2., -3.]},
           "program": target.to_dict(), "program_equivalent": True,
           "answer_correct": True}
    example = SimpleNamespace(source_text=source, program=target, inputs=(2, 3))
    small = replay_budget(row, example, plan, max_nodes=4)
    assert small["scored_decisions"] < len(transcript)
    assert small["requested_top_k_proven"] is False
    complete = replay_budget(row, example, plan, max_nodes=256)
    assert complete["procedure_equivalent"] is True
    assert complete["answer_correct"] is True
    assert complete["scored_decisions"] == len(transcript)
    row["search"]["score_transcript"][0]["choices"] = ["wrong"]
    with pytest.raises(ValueError, match="choice inventory"):
        replay_budget(row, example, plan, max_nodes=4)
