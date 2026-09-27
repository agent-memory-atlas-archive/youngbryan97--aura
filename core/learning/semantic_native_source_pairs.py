"""Bind witnessed source contrasts to the native grammar's first divergent choice."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from core.learning.semantic_graph_counterexamples import (
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_native_decision_supervision import native_teacher_decisions

if TYPE_CHECKING:
    import mlx.core as mx

    from core.learning.semantic_program_ir import SemanticValue
    from core.learning.semantic_program_transducer import SemanticTransducerTrainingExample

SOURCE_PAIR_CONTRACT = {
    "basis": "fit_only_shared_contrast_lineage_and_witnessed_meaning_change",
    "decision": "first_shared_typed_grammar_choice_with_opposite_targets",
    "loss": "logistic_source_by_choice_interaction",
    "weight": 1.0,
    "calibration": "ordinary_conditional_grammar_choice_loss",
    "held_labels_used": False,
}


def _input_types(values: Iterable[SemanticValue]) -> tuple[str, ...]:
    return tuple("integer_sequence" if isinstance(value, tuple) else "integer"
                 for value in values)


def _first_divergence(left: SemanticTransducerTrainingExample,
                      right: SemanticTransducerTrainingExample,
                      register_encoding: str) -> dict[str, int | str] | None:
    if _input_types(left.public_inputs) != _input_types(right.public_inputs):
        return None
    left_decisions = native_teacher_decisions(left.ir.to_program(),
        _input_types(left.public_inputs), register_encoding=register_encoding)
    right_decisions = native_teacher_decisions(right.ir.to_program(),
        _input_types(right.public_inputs), register_encoding=register_encoding)
    if len(left_decisions) != len(right_decisions):
        return None
    for ordinal, (a, b) in enumerate(zip(left_decisions, right_decisions, strict=True)):
        values = tuple(choice.value for choice in a.choices)
        if a.kind != b.kind or values != tuple(choice.value for choice in b.choices):
            return None
        if a.correct_index != b.correct_index:
            return {"decision_index": ordinal, "kind": a.kind,
                    "own_index": a.correct_index, "partner_index": b.correct_index}
    return None


def native_source_pair_plan(examples: Iterable[SemanticTransducerTrainingExample],
                            fit_ids: Iterable[str], *,
                            register_encoding: str) -> dict[str, dict[str, int | str]]:
    """Select only fit-local contrasts whose denotations and next choices differ.

    Contrast lineage, type admission, and an execution witness establish the
    pair. Token overlap only chooses the least changed source among valid peers;
    it is never a model feature or a correctness label.
    """
    fit_ids = set(fit_ids)
    by_id = {item.ir.source_text_sha256: item for item in examples
             if item.ir.source_text_sha256 in fit_ids}
    if len(by_id) != len(fit_ids) or any(item.split != "train" for item in by_id.values()):
        raise ValueError("native source pairs require the complete fit-only partition")
    grouped = defaultdict(list)
    for identity, item in by_id.items():
        if item.contrast_id:
            grouped[item.contrast_id].append((identity, item))
    pairs = {}
    for members in grouped.values():
        for identity, item in members:
            options = []
            for partner_id, partner in members:
                if identity == partner_id:
                    continue
                divergence = _first_divergence(item, partner, register_encoding)
                if divergence is None:
                    continue
                comparison = compare_program_meanings(item.ir.to_program(),
                    partner.ir.to_program(), counterfactual_inputs(item.public_inputs))
                if comparison["status"] != "different" or comparison.get("witness") is None:
                    continue
                overlap = SequenceMatcher(None, tuple(item.ir.source_token_ids),
                    tuple(partner.ir.source_token_ids), autojunk=False).ratio()
                options.append((overlap, partner_id, divergence))
            if options:
                _overlap, partner_id, divergence = min(options,
                    key=lambda option: (-option[0], option[1]))
                pairs[identity] = {"partner": partner_id, **divergence,
                    "selection": "same_contrast_witnessed_first_divergence_max_token_overlap"}
    return dict(sorted(pairs.items()))


def native_source_interaction_loss(own_correct: mx.array, own_rival: mx.array,
                                   partner_correct: mx.array,
                                   partner_rival: mx.array) -> mx.array:
    """A logistic loss on the source-by-choice interaction, cancelling wire bias."""
    import mlx.core as mx

    interaction = (own_correct - own_rival) - (partner_correct - partner_rival)
    return mx.logaddexp(mx.array(0., dtype=mx.float32), -interaction)
