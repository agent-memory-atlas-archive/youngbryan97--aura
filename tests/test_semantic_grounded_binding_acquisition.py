"""Grounded acquisition and source training have separate access to labels."""

from dataclasses import replace

import mlx.core as mx
import pytest

from core.learning.semantic_grounded_binding_acquisition import (
    grounded_evidence_from_source_example,
    grounded_source_equivariance_pairs,
    grounded_supervision_from_source_example,
)
from tests.test_semantic_program_shared_transducer import _shared_example


def example(variant=0):
    return _shared_example(three_steps=False, variant=variant, split="train")


def test_inference_evidence_does_not_change_when_teacher_argument_registers_change():
    item = example()
    changed = replace(item, ir=replace(item.ir, instructions=tuple(
        replace(instruction, args=tuple(reversed(instruction.args))) for instruction in item.ir.instructions)))
    original, _ = grounded_evidence_from_source_example(item)
    altered, _ = grounded_evidence_from_source_example(changed)
    assert original.receipt() == altered.receipt()
    for first, second in zip(original.arrays()[0], altered.arrays()[0], strict=True):
        assert mx.array_equal(first, second).item()
    with pytest.raises(ValueError, match="test targets"):
        grounded_supervision_from_source_example(replace(item, split="test"))


def test_graph_supervision_uses_witnessed_program_equivalence_and_computed_candidates():
    item = example()
    supervision = grounded_supervision_from_source_example(item, equivalence_supervision=True)
    supervision.indices()
    assert supervision.graphs and supervision.positive_graphs
    assert any(record.identity.startswith("register:3") for record in supervision.evidence.context.referents)
    assert supervision.evidence.roles[-2].role.startswith("sub:")
    assert supervision.positives[-2] == ((item.ir.source_text_sha256, "register:3"),)
    original = tuple(positive[0] for positive in grounded_supervision_from_source_example(item).positives)
    swap = (*original[:-2], original[-1], original[-2])
    assert swap in supervision.graphs and supervision.graphs.index(swap) not in supervision.positive_graphs


def test_source_equivariance_requires_exact_fit_cohort_and_real_correspondence():
    items = (example(0), example(2))
    examples = tuple(grounded_supervision_from_source_example(item) for item in items)
    pairs = grounded_source_equivariance_pairs(items, examples)
    assert len(pairs) == 1 and pairs[0].left != pairs[0].right
    pairs[0].orders({item.evidence.source_id: item for item in examples})
    with pytest.raises(ValueError, match="exact source fit"):
        grounded_source_equivariance_pairs((replace(items[0], split="validation"), items[1]), examples)
