"""Training choices match inference, including roles, branches, and explicit stop."""

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_codec import REGISTER_ENCODINGS
from core.learning.semantic_native_decision_supervision import (
    GRAMMAR_CHOICE_CONTRACT,
    native_decision_choice_loss,
    native_teacher_decisions,
)
from core.learning.semantic_native_path_objective import (
    GRAMMAR_PATH_CONTRACT,
    JOINT_GRAPH_CONTRAST_CONTRACT,
    path_choice_contract,
)
from core.learning.semantic_native_path_selection import JOINT_GRAPH_SELECTION_CONTRACT
from core.learning.semantic_native_source_control import (
    SOURCE_ERASURE_CONTRACT,
    source_control_mode_from_plan,
)
from tests.test_semantic_native_program import Tokenizer
from tools.train_semantic_native_program import (
    build_native_supervision,
    native_grammar_source_loss,
    native_grammar_supervision_sets,
    native_fit_schema_version,
    native_loss,
)
from tools.verify_semantic_native_fit import verify_source_control_supervision


@pytest.mark.parametrize("encoding", REGISTER_ENCODINGS)
def test_operation_and_finish_training_uses_every_inference_alternative(encoding):
    program = Program(2, (Instruction("sub", (1, 0)),))
    groups = native_teacher_decisions(program, ("integer",) * 2, register_encoding=encoding)
    assert [row.kind for row in groups] == ["operation", "reference", "reference", "termination"]
    assert "mul" in [choice.value for choice in groups[0].choices]
    assert groups[0].choices[groups[0].correct_index].value == "sub"
    assert [choice.value for choice in groups[-1].choices] == ["finish", "continue"]
    assert groups[-1].correct_index == 0
    references = [row.choices[row.correct_index].value for row in groups[1:3]]
    assert references == ([1, 0] if encoding == "absolute_v1" else ["input:1", "input:0"])


def test_numeric_registers_are_whole_atoms_not_prefix_matches():
    program = Program(11, (Instruction("sub", (10, 1)),))
    groups = native_teacher_decisions(program, ("integer",) * 11)
    assert groups[1].choices[groups[1].correct_index].value == 10
    assert groups[2].choices[groups[2].correct_index].value == 1


def test_disconnected_branch_can_only_continue_until_join_then_learns_finish():
    program = Program(4, (Instruction("sub", (0, 1)), Instruction("mul", (2, 3)),
                          Instruction("add", (4, 5))))
    groups = native_teacher_decisions(program, ("integer",) * 4)
    stops = [row for row in groups if row.kind == "termination"]
    assert [[choice.value for choice in row.choices] for row in stops] == [
        ["finish", "continue"], ["continue"], ["finish", "continue"]]
    assert [row.choices[row.correct_index].value for row in stops] == ["continue", "continue", "finish"]


def test_public_type_constraints_remain_the_same_for_unary_and_sequence_choices():
    program = Program(2, (Instruction("at", (0, 1)), Instruction("neg", (2,))))
    groups = native_teacher_decisions(program, ("integer_sequence", "integer"))
    assert [choice.value for choice in groups[1].choices] == [0]
    assert groups[-1].choices[groups[-1].correct_index].value == "finish"
    with pytest.raises(ValueError, match="admitted"):
        native_teacher_decisions(program, ("integer",) * 2)
    with pytest.raises(ValueError, match="public types"):
        native_teacher_decisions(program, ("integer",))


@pytest.mark.parametrize("count", [1, 2, 18])
def test_choice_gradient_is_exact_conditional_probability_and_zero_when_deterministic(count):
    import mlx.core as mx

    scores = mx.arange(count).astype(mx.float32) / 10.
    correct = count - 1
    loss = native_decision_choice_loss(scores, correct)
    expected = mx.logsumexp(scores) - scores[correct]
    gradient = mx.grad(lambda value: native_decision_choice_loss(value, correct))(scores)
    assert mx.allclose(loss, expected, atol=1e-6).item()
    one_hot = mx.array([float(index == correct) for index in range(count)])
    assert mx.allclose(gradient, mx.softmax(scores) - one_hot, atol=1e-6).item()
    assert mx.allclose(native_decision_choice_loss(scores + 7., correct), loss, atol=1e-6).item()


def fixture(*, erase=False):
    tokenizer = Tokenizer()
    sources = ("Subtract 2 from 5.", "Take 3 away from 7.")
    texts = {hashlib.sha256(text.encode()).hexdigest(): text for text in sources}
    target = Program(2, (Instruction("sub", (1, 0)),))
    items = {identity: SimpleNamespace(split="train", public_inputs=values, ir=SimpleNamespace(
        source_text_sha256=identity, source_token_ids=tuple(tokenizer.encode(text)),
        to_program=lambda: target)) for (identity, text), values in zip(
            texts.items(), ((2, 5), (3, 7)), strict=True)}
    fit, cal = tuple(sorted(items))
    sequences, groups, rows = native_grammar_supervision_sets(
        items, texts, tokenizer, tuple(items), register_encoding="role_relative_v1",
        source_erasure_ids=(fit,) if erase else ())
    plan = {"schema": "aura.semantic_native_fit_plan.v3", "objective": "grammar_choices",
        "loss_scope": "semantic_decisions", "grammar_choice_contract": GRAMMAR_CHOICE_CONTRACT,
        "fit_ids": [fit], "captured_fit_ids": [fit], "calibration_ids": [cal],
        "max_sequence_tokens": 1024, "register_encoding": "role_relative_v1"}
    receipt = {"rows": rows, "grammar_choice_contract": GRAMMAR_CHOICE_CONTRACT}
    if erase:
        plan["source_evidence_control"] = SOURCE_ERASURE_CONTRACT
        receipt["source_evidence_control"] = {**SOURCE_ERASURE_CONTRACT,
            "erased_fit_ids": [fit], "unchanged_calibration_ids": [cal]}
    return plan, json.loads(json.dumps(receipt)), items, texts, tokenizer, sequences, groups


@pytest.mark.parametrize("erase", [False, True])
def test_independent_verifier_rebuilds_every_choice_and_every_token(erase):
    plan, receipt, items, _texts, tokenizer, sequences, groups = fixture(erase=erase)
    result = verify_source_control_supervision(plan, receipt, items, tokenizer)
    assert result["grammar_choices_verified"] is True
    assert result["source_control_verified"] is erase
    assert result["supervision_sequences_verified"] == len(sequences)
    assert result["supervised_decisions_verified"] == sum(len(value) for value in groups.values())


@pytest.mark.parametrize("field", ["choice", "correct_index", "kind", "span", "tokens", "duplicate", "missing"])
def test_choice_or_token_forgery_cannot_enter_verified_training(field):
    plan, receipt, items, _texts, tokenizer, _sequences, _groups = fixture()
    if field == "duplicate":
        receipt["rows"].append(deepcopy(receipt["rows"][0]))
    elif field == "missing":
        receipt["rows"].pop()
    else:
        receipt["rows"][0][field] = "forged"
    with pytest.raises(ValueError, match="reconstruction"):
        verify_source_control_supervision(plan, receipt, items, tokenizer)


def test_validation_targets_and_nonpublic_register_order_are_refused():
    _plan, _receipt, items, texts, tokenizer, _sequences, _groups = fixture()
    key = next(iter(items))
    items[key].split = "validation"
    with pytest.raises(ValueError, match="public source-order"):
        native_grammar_supervision_sets(items, texts, tokenizer, tuple(items))
    items[key].split = "train"
    items[key].public_inputs = tuple(reversed(items[key].public_inputs))
    with pytest.raises(ValueError, match="public source-order"):
        native_grammar_supervision_sets(items, texts, tokenizer, tuple(items))


def test_new_objective_cannot_relabel_an_archived_fit():
    plan, _receipt, _items, _texts, _tokenizer, _sequences, _groups = fixture()
    assert source_control_mode_from_plan(plan) == "source_text"
    for schema in ("aura.semantic_native_fit_plan.v1", "aura.semantic_native_fit_plan.v2"):
        with pytest.raises(ValueError, match="historical"):
            source_control_mode_from_plan({**plan, "schema": schema})
    with pytest.raises(ValueError, match="grammar-choice"):
        source_control_mode_from_plan({**plan, "grammar_choice_contract": {}})


def test_joint_graph_training_reuses_source_bound_grammar_and_reconstructs_all_rows(monkeypatch):
    from tools.semantic_native_execution import execution_contract

    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    plan, _old, items, texts, tokenizer, _sequences, _groups = fixture()
    plan.update(schema="aura.semantic_native_fit_plan.v7",
                grammar_choice_contract=path_choice_contract(),
                grammar_path_objective_contract=GRAMMAR_PATH_CONTRACT,
                path_checkpoint_selection_contract=JOINT_GRAPH_SELECTION_CONTRACT,
                selection="baseline_preserving_joint_source_calibration",
                unfitted_checkpoint_eligible=True,
                joint_graph_contrast_limit=3,
                graph_contrast_contract=JOINT_GRAPH_CONTRAST_CONTRACT,
                fit_ids=plan["fit_ids"],
                prefix_storage_contract={"mode": "source_shards"},
                execution_contract=execution_contract(precision="float32", prefix_strategy="trie"),
                plan_sha256="a" * 64)
    peers = tuple(items[key].ir.to_program() for key in plan["fit_ids"])
    sequences, decisions, graphs, receipt = build_native_supervision(
        items, texts, tokenizer, tuple(sorted(items)), plan, peers)
    assert set(graphs) == set(items)
    assert len(sequences) == len(receipt["rows"]) + len(receipt["graph_rows"])
    assert all(keys[0][1:] == (-1, 0) for keys in graphs.values())
    assert all(len(keys) >= 2 for keys in graphs.values())
    assert all(receipt["graph_rows"][index]["positive"] for index in
               (0, len(graphs[tuple(sorted(items))[0]])))
    checked = verify_source_control_supervision(plan, json.loads(json.dumps(receipt)), items, tokenizer)
    assert checked["supervision_sequences_verified"] == len(sequences)
    assert checked["whole_graph_sequences_verified"] == len(receipt["graph_rows"])
    assert checked["supervised_decisions_verified"] == sum(map(len, decisions.values()))
    forged = json.loads(json.dumps(receipt))
    forged["graph_rows"][0]["program_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="whole-graph supervision"):
        verify_source_control_supervision(plan, forged, items, tokenizer)


@pytest.mark.parametrize(("objective", "typed_pairs", "graph_contrasts", "expected"), [
    ("grammar_choices", False, 0, 5),
    ("grammar_source_pairs", True, 0, 6),
    ("grammar_choices", False, 3, 7),
    ("grammar_source_pairs", True, 3, 7),
])
def test_composed_objectives_share_one_plan_and_report_schema(
        objective, typed_pairs, graph_contrasts, expected):
    assert native_fit_schema_version(
        objective=objective, source_evidence="source_text", path_objective=True,
        typed_pairs=typed_pairs, joint_graph_contrasts=graph_contrasts) == expected


def test_real_mlx_source_objective_updates_the_scores_used_at_inference():
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    _plan, _receipt, _items, _texts, _tokenizer, sequences, groups = fixture()
    decisions = groups[next(iter(groups))]
    keys, correct = decisions[0]

    class Suffix(nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = mx.zeros((len(keys), 256))

        def __call__(self, hidden, *, logit_positions):
            row = mx.take(self.logits, hidden[0, 0, 0].astype(mx.int32), axis=0)
            return mx.broadcast_to(row, (1, len(logit_positions), 256))

    tail = Suffix()
    states = {key: mx.full((1, len(sequences[key].tokens) - 1, 1), index)
              for index, key in enumerate(keys)}
    baseline = native_grammar_source_loss(tail, states, sequences, ((keys, correct),)).item()
    optimizer = optim.SGD(learning_rate=.5)
    for _ in range(8):
        loss, gradients = nn.value_and_grad(tail, lambda suffix: native_grammar_source_loss(
            suffix, states, sequences, ((keys, correct),)))(tail)
        optimizer.update(tail, gradients)
        mx.eval(tail.parameters(), optimizer.state, loss)
    after = native_grammar_source_loss(tail, states, sequences, ((keys, correct),)).item()
    assert after < baseline
    scores = tuple(-native_loss(tail, states[key], sequences[key], summed=True,
                               scope="semantic_decisions").item() for key in keys)
    assert max(range(len(scores)), key=scores.__getitem__) == correct
