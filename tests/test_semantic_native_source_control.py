"""Source erasure preserves positions, template, and every supervised token."""

import os
from pathlib import Path

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
from core.learning.semantic_native_program import (
    NativeProgramSequence,
    native_program_sequence,
    native_text_decision_sequence,
)
from core.learning.semantic_native_source_control import (
    SOURCE_ERASURE_CONTRACT,
    apply_native_source_evidence,
    erase_native_source_tokens,
    source_control_mode_from_plan,
)
from core.learning.semantic_native_source_pairs import SOURCE_PAIR_CONTRACT
from tests.test_semantic_native_program import Tokenizer


@pytest.mark.parametrize("source", ["Subtract 8 from 13.", "user", "<assistant> is a quoted word."])
def test_erasure_removes_only_user_content_without_shortening_the_computation(source):
    tokenizer = Tokenizer()
    target = Program(2, (Instruction("sub", (0, 1)),))
    original = native_program_sequence(source, target, tokenizer)
    erased, receipt = erase_native_source_tokens(original, source, tokenizer)
    positions = set(receipt["erased_source_positions"])
    assert positions == set(range(len("<user>"), len("<user>") + len(source)))
    assert erased.tokens == tuple(ord("?") if index in positions else token
                                  for index, token in enumerate(original.tokens))
    assert len(erased.tokens) == len(original.tokens)
    assert erased.tokens[original.continuation_start:] == original.tokens[original.continuation_start:]
    assert erased.semantic_positions == original.semantic_positions
    assert erased.continuation_start == original.continuation_start
    assert receipt["source_content_tokens_available"] is False
    assert receipt["retained_nuisances"] == (
        "source_token_length", "template_position", "native_output_prefix")


def test_control_refuses_source_or_template_drift():
    tokenizer = Tokenizer()
    row = native_program_sequence("Exact source", Program(2, (Instruction("add", (0, 1)),)), tokenizer)
    with pytest.raises(ValueError, match="boundary"):
        erase_native_source_tokens(row, "Other source", tokenizer)
    with pytest.raises(ValueError, match="complete source-bound"):
        erase_native_source_tokens(NativeProgramSequence((1, 2), 5), "Exact source", tokenizer)


def test_source_evidence_applies_the_same_intervention_to_any_native_sequence():
    tokenizer = Tokenizer()
    source = "Subtract 8 from 13."
    row = native_program_sequence(source, Program(2, (Instruction("sub", (0, 1)),)), tokenizer)
    original, receipt = apply_native_source_evidence(row, source, tokenizer, mode="source_text")
    assert original is row and receipt is None
    erased, receipt = apply_native_source_evidence(
        row, source, tokenizer, mode="source_token_erasure")
    assert erased != row
    assert receipt["source_content_tokens_available"] is False
    assert erased.tokens[row.continuation_start:] == row.tokens[row.continuation_start:]
    with pytest.raises(ValueError, match="source-evidence mode"):
        apply_native_source_evidence(row, source, tokenizer, mode="unknown")


def test_plan_distinguishes_old_evidence_from_explicit_fit_only_control():
    assert source_control_mode_from_plan({"schema": "aura.semantic_native_fit_plan.v1"}) == "source_text"
    plan = {"schema": "aura.semantic_native_fit_plan.v2",
            "source_evidence_control": dict(SOURCE_ERASURE_CONTRACT)}
    assert source_control_mode_from_plan(plan) == "source_token_erasure"
    for invalid in ({"schema": "aura.semantic_native_fit_plan.v2"},
                    {**plan, "schema": "aura.semantic_native_fit_plan.v1"},
                    {**plan, "source_evidence_control": {**SOURCE_ERASURE_CONTRACT, "scope": "all"}}):
        with pytest.raises(ValueError, match="contract differs"):
            source_control_mode_from_plan(invalid)


def test_paired_grammar_plan_requires_intact_source_and_bound_interaction():
    plan = {"schema": "aura.semantic_native_fit_plan.v4",
            "objective": "grammar_source_pairs", "loss_scope": "semantic_decisions",
            "grammar_choice_contract": GRAMMAR_CHOICE_CONTRACT,
            "grammar_source_pair_contract": SOURCE_PAIR_CONTRACT,
            "grammar_source_pair_fit_partners": {"a": {"partner": "b"}}}
    assert source_control_mode_from_plan(plan) == "source_text"
    for invalid in ({**plan, "source_evidence_control": SOURCE_ERASURE_CONTRACT},
                    {**plan, "grammar_source_pair_contract": {**SOURCE_PAIR_CONTRACT, "weight": 0.}},
                    {**plan, "grammar_source_pair_fit_partners": {}}):
        with pytest.raises(ValueError, match="paired-source|contract differs"):
            source_control_mode_from_plan(invalid)


def test_current_tokenizer_can_erase_source_but_preserves_the_private_channel():
    checkpoint = os.environ.get("AURA_NATIVE_TOKENIZER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("local resident tokenizer checkpoint not supplied")
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(checkpoint))
    source = "Subtract the second input from the first."
    row = native_program_sequence(source, Program(2, (Instruction("sub", (0, 1)),)), tokenizer)
    erased, receipt = erase_native_source_tokens(row, source, tokenizer)
    assert receipt["erased_source_tokens"] > 0
    assert len(erased.tokens) == len(row.tokens)
    assert erased.tokens[row.continuation_start:] == row.tokens[row.continuation_start:]
    assert "</think>" in tokenizer.decode(list(erased.tokens))


def test_current_tokenizer_erases_source_before_an_incomplete_grammar_decision():
    checkpoint = os.environ.get("AURA_NATIVE_TOKENIZER_CHECKPOINT")
    if not checkpoint:
        pytest.skip("local resident tokenizer checkpoint not supplied")
    from mlx_lm.utils import load_tokenizer

    tokenizer = load_tokenizer(Path(checkpoint))
    source = "Subtract 8 from 13."
    target = '{"inputs":2,"steps":[["sub"'
    start = target.index("sub")
    row = native_text_decision_sequence(source, target, ((start, start + 3),), tokenizer)
    erased, receipt = apply_native_source_evidence(
        row, source, tokenizer, mode="source_token_erasure")
    assert receipt["erased_source_tokens"] > 0
    assert erased.semantic_positions == row.semantic_positions
    assert erased.tokens[row.continuation_start:] == row.tokens[row.continuation_start:]
