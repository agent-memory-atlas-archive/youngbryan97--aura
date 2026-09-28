"""Conditional choices must not be counted as independent successful programs."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from core.learning.frozen_prefix_branches import native_source_anchor
from core.learning.semantic_native_grammar import decode_native_grammar
from core.learning.semantic_native_program import native_text_decision_sequence
from core.learning.semantic_native_relative_program import REGISTER_ENCODING
from core.learning.semantic_native_source_control import native_score_input_receipt
from tests.test_semantic_native_program import Tokenizer
from tools.probe_semantic_native_guided_binding import guided_choice_scores
from tools.verify_semantic_native_guided_binding import replay_guided_row


def test_guide_keeps_a_disagreeing_base_winner_visible():
    choices = [SimpleNamespace(value="add"), SimpleNamespace(value="sub")]
    guide = {"choices": ["add", "sub"], "chosen": "sub"}
    measured = [2., 1.]
    assert guided_choice_scores(choices, guide, measured) == (0., 1.)
    assert measured == [2., 1.]


@pytest.mark.parametrize("scores", [[1.], [float("nan"), 0.], [0., float("inf")], [True, 0.]])
def test_guide_refuses_missing_or_nonfinite_scores(scores):
    with pytest.raises(ValueError, match="differ"):
        guided_choice_scores([SimpleNamespace(value="add"), SimpleNamespace(value="sub")],
                             {"choices": ["add", "sub"], "chosen": "sub"}, scores)


@pytest.mark.parametrize("guide", [{"choices": ["sub", "add"], "chosen": "sub"},
                                  {"choices": ["add", "sub"], "chosen": "mul"}])
def test_guide_refuses_changed_inventory_or_unsupported_winner(guide):
    with pytest.raises(ValueError, match="differ"):
        guided_choice_scores([SimpleNamespace(value="add"), SimpleNamespace(value="sub")], guide, [1., 0.])


def fixture_row():
    tokenizer = Tokenizer()
    example = SimpleNamespace(source_text="Add 8 and 3.", inputs=(8, 3))
    receipts, anchors = [], []

    def score(choices):
        sequences = tuple(native_text_decision_sequence(
            example.source_text, choice.text, (choice.span,), tokenizer, max_tokens=1024)
            for choice in choices)
        receipts.append([native_score_input_receipt(sequence, None) for sequence in sequences])
        anchors.append(native_source_anchor(sequences))
        return tuple(float(choice.value in ("add", "input:0", "finish")) for choice in choices)

    decoded = decode_native_grammar(("integer", "integer"), score, register_encoding=REGISTER_ENCODING)
    trace = json.loads(json.dumps(decoded.trace))
    decisions = [{"kind": entry["kind"], "choices": entry["choices"],
                  "guide_chosen": entry["chosen"], "base_scores": entry["scores"],
                  "base_winner": entry["chosen"]} for entry in trace]
    first = decisions[0]
    wrong = next(index for index, value in enumerate(first["choices"]) if value != first["guide_chosen"])
    first["base_scores"] = [float(index == wrong) for index in range(len(first["choices"]))]
    first["base_winner"] = first["choices"][wrong]
    assert all(anchor == anchors[0] for anchor in anchors)
    row = {"guide_row_receipt_sha256": "guide", "decisions": decisions,
           "score_input_receipts": receipts,
           "prefix_execution": {"schema": "aura.frozen_prefix_branches.v2",
                                "suffix_computation_unchanged": True,
                                "anchor_tokens": len(anchors[0]),
                                "anchor_token_sha256": hashlib.sha256(
                                    json.dumps(anchors[0], separators=(",", ":")).encode("ascii")).hexdigest(),
                                "trie_calls": len(receipts),
                                "branches": sum(map(len, receipts))}}
    guide = {"receipt_sha256": "guide", "decision_trace": trace, "program": decoded.program.to_dict()}
    plan = {"schema": "aura.semantic_native_grammar_plan.v7", "prefix_strategy": "trie",
            "max_steps": 8, "register_encoding": REGISTER_ENCODING}
    return row, guide, example, plan, tokenizer


def test_replay_counts_base_disagreement_despite_exact_guided_graph():
    row, guide, example, plan, tokenizer = fixture_row()
    totals = replay_guided_row(row, guide, example=example, guide_plan=plan,
                              tokenizer=tokenizer, max_tokens=1024)
    assert totals["operation"] == {"matched": 0, "total": 1}
    assert totals["reference"] == {"matched": 2, "total": 2}
    assert totals["termination"] == {"matched": 1, "total": 1}


@pytest.mark.parametrize("mutation", ["winner", "kind", "guide", "tokens", "anchor", "population", "scores"])
def test_replay_rejects_tampered_conditional_evidence(mutation):
    row, guide, example, plan, tokenizer = fixture_row()
    row = copy.deepcopy(row)
    if mutation == "winner":
        row["decisions"][0]["base_winner"] = row["decisions"][0]["guide_chosen"]
    elif mutation == "kind":
        row["decisions"][0]["kind"] = "reference"
    elif mutation == "guide":
        row["guide_row_receipt_sha256"] = "other"
    elif mutation == "tokens":
        row["score_input_receipts"][0][0]["sequence_sha256"] = "changed"
    elif mutation == "anchor":
        row["prefix_execution"]["anchor_token_sha256"] = "changed"
    elif mutation == "population":
        row["decisions"].pop()
    else:
        row["decisions"][0]["base_scores"][0] = float("nan")
    with pytest.raises(ValueError):
        replay_guided_row(row, guide, example=example, guide_plan=plan,
                          tokenizer=tokenizer, max_tokens=1024)
