"""Witnessed fit-only contrasts, not construction labels, drive source interactions."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_source_pairs import (
    native_source_interaction_loss,
    native_source_pair_plan,
)


def example(identity, contrast, program, *, tokens, split="train"):
    return SimpleNamespace(ir=SimpleNamespace(source_text_sha256=identity,
                           source_token_ids=tokens, to_program=lambda: program),
                           public_inputs=(8, 3), contrast_id=contrast, split=split)


def test_pair_plan_uses_only_witnessed_fit_local_first_divergences():
    add = Program(2, (Instruction("add", (0, 1)),))
    sub = Program(2, (Instruction("sub", (0, 1)),))
    rows = (example("a", "lineage", add, tokens=(1, 2, 3, 4)),
            example("b", "lineage", sub, tokens=(1, 2, 5, 4)),
            example("c", "lineage", add, tokens=(7, 8, 9, 10)),
            example("held", "lineage", sub, tokens=(1, 2, 6, 4), split="validation"))
    pairs = native_source_pair_plan(rows, ("a", "b", "c"), register_encoding="absolute_v1")
    assert pairs["a"]["partner"] == "b"
    assert pairs["a"]["kind"] == "operation"
    assert pairs["a"]["own_index"] != pairs["a"]["partner_index"]
    assert pairs["b"]["partner"] == "a"
    assert pairs["c"]["partner"] == "b"
    assert "held" not in pairs
    with pytest.raises(ValueError, match="complete fit-only"):
        native_source_pair_plan(rows, ("a", "held"), register_encoding="absolute_v1")


def test_pair_plan_refuses_unchanged_meaning():
    add = Program(2, (Instruction("add", (0, 1)),))
    rows = (example("a", "same", add, tokens=(1, 2)),
            example("b", "same", add, tokens=(3, 4)))
    assert native_source_pair_plan(rows, ("a", "b"), register_encoding="absolute_v1") == {}


def test_pair_plan_admits_source_bound_reference_contrast():
    left = Program(2, (Instruction("sub", (0, 1)),))
    right = Program(2, (Instruction("sub", (1, 0)),))
    rows = (example("a", "role", left, tokens=(1, 2, 3)),
            example("b", "role", right, tokens=(1, 4, 3)))
    pairs = native_source_pair_plan(rows, ("a", "b"), register_encoding="absolute_v1")
    assert pairs["a"]["kind"] == pairs["b"]["kind"] == "reference"
    assert pairs["a"]["own_index"] == pairs["b"]["partner_index"]


def test_pair_plan_admits_shared_prefix_termination_contrast():
    short = Program(2, (Instruction("add", (0, 1)),))
    long = Program(2, (Instruction("add", (0, 1)), Instruction("mul", (2, 1))))
    rows = (example("a", "depth", short, tokens=(1, 2, 3)),
            example("b", "depth", long, tokens=(1, 2, 4)))
    pairs = native_source_pair_plan(rows, ("a", "b"), register_encoding="absolute_v1")
    assert pairs["a"]["kind"] == pairs["b"]["kind"] == "termination"
    assert pairs["a"]["own_index"] != pairs["a"]["partner_index"]


def test_interaction_loss_requires_the_preference_to_change_with_source():
    equal = native_source_interaction_loss(mx.array(2.), mx.array(0.),
                                           mx.array(2.), mx.array(0.)).item()
    flipped = native_source_interaction_loss(mx.array(2.), mx.array(0.),
                                             mx.array(0.), mx.array(2.)).item()
    assert flipped < equal
    parameter = mx.array(0.)
    derivative = mx.grad(lambda value: native_source_interaction_loss(
        value, mx.array(0.), mx.array(0.), mx.array(0.)))(parameter)
    assert derivative.item() < 0
