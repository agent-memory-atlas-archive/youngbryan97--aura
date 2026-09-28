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


@pytest.mark.parametrize("version", ["v7", "v8"])
def test_replay_counts_base_disagreement_despite_exact_guided_graph(version):
    row, guide, example, plan, tokenizer = fixture_row()
    plan["schema"] = f"aura.semantic_native_grammar_plan.{version}"
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


@pytest.mark.parametrize("mutation", [None, "counts", "population", "authority", "guide", "row", "extra"])
def test_document_replay_pins_guide_population_and_report(tmp_path, monkeypatch, mutation):
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.verify_semantic_native_guided_binding import verify_guided

    row, guide, example, guide_plan, tokenizer = fixture_row()
    identity = hashlib.sha256(example.source_text.encode()).hexdigest()
    guide_plan.update(plan_sha256="guide-plan", sources=[identity], dataset="role_intervention", seed=0)
    guide_report = {"receipt_sha256": "guide-report", "row_receipts": {identity: "guide"}}
    training = {"plan_sha256": "training", "model_descriptor_sha256": "model",
                "pointer_sha256": "pointer", "model_path": "/unused/model", "max_sequence_tokens": 1024}
    selected = {"receipt_sha256": "checkpoint"}
    monkeypatch.setattr("tools.probe_semantic_native_guided_binding.guide_basis",
                        lambda *_, **__: (training, selected, guide_plan, guide_report, (guide,)))
    monkeypatch.setattr("tools.verify_semantic_native_grammar.verified_examples", lambda *_, **__: (example,))
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda *_: tokenizer)
    body = {"schema": "aura.native_guided_binding_plan.v1",
            "training_plan_sha256": "training", "checkpoint_receipt_sha256": "checkpoint",
            "guide_plan_sha256": "guide-plan", "guide_report_receipt_sha256": "guide-report",
            "guide_row_receipts": guide_report["row_receipts"],
            "model_descriptor_sha256": "model", "pointer_sha256": "pointer",
            "dataset": "role_intervention", "seed": 0, "sources": [identity],
            "guidance": "fitted_target_blind_trace_all_prior_choices",
            "cohort_selection": "all_exact_fitted_guide_rows", "measured_weight_mode": "base",
            "precision": "float32", "prefix_strategy": "trie", "target_available_to_scorer": False,
            "serving_authority": False, "qualification_evidence": False, "implementation": {}}
    plan = {**body, "plan_sha256": digest(body)}
    row.update(schema="aura.native_guided_binding_row.v1", source_sha256=identity,
               plan_sha256=plan["plan_sha256"])
    if mutation == "row":
        row["source_sha256"] = "wrong"
    row["receipt_sha256"] = digest(row)
    counts = {"operation": {"matched": 0, "total": 1}, "reference": {"matched": 2, "total": 2},
              "termination": {"matched": 1, "total": 1}}
    report = {"schema": "aura.native_guided_binding.v1", "plan_sha256": plan["plan_sha256"],
              "population": 1, "row_receipts": {identity: row["receipt_sha256"]},
              "guidance_is_diagnostic_only": True, "serving_authority": False,
              "qualification_evidence": False, "base_matches_fitted_by_kind": counts}
    if mutation == "counts":
        report["base_matches_fitted_by_kind"]["operation"]["matched"] = 1
    elif mutation == "population":
        report["population"] = 2
    elif mutation == "authority":
        report["serving_authority"] = True
    elif mutation == "guide":
        report["guidance_is_diagnostic_only"] = False
    report["receipt_sha256"] = digest(report)
    (tmp_path / "rows").mkdir()
    for name, document in (("plan.json", plan), ("report.json", report), (f"rows/{identity}.json", row)):
        (tmp_path / name).write_text(json.dumps(document))
    if mutation == "extra":
        (tmp_path / "rows/extra.json").write_text("{}")
    if mutation:
        with pytest.raises(ValueError):
            verify_guided(tmp_path, tmp_path, tmp_path)
    else:
        result = verify_guided(tmp_path, tmp_path, tmp_path)
        assert result["base_matches_fitted_by_kind"] == counts
        assert result["model_scores_recomputed"] is False
        assert result["end_to_end_gain_proven"] is False
        assert result["general_transfer_proven"] is False
