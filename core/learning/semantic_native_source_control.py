"""Erase source content without changing native template or target positions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from core.learning.semantic_native_program import NativeProgramSequence

SOURCE_ERASURE_CONTRACT = {
    "mode": "source_token_erasure",
    "scope": "fit_only_calibration_source_unchanged",
    "filler_text": "?",
    "sequence_length_unchanged": True,
    "template_tokens_unchanged": True,
    "supervision_tokens_unchanged": True,
    "retained_nuisances": ["source_token_length", "template_position", "native_output_prefix"],
}


def source_control_mode_from_plan(plan: Mapping[str, Any]) -> str:
    """Keep historical fits intact and require explicit authority for erasure."""
    schema = plan.get("schema")
    if schema in {"aura.semantic_native_fit_plan.v3", "aura.semantic_native_fit_plan.v4"}:
        from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
        from core.learning.semantic_native_source_pairs import SOURCE_PAIR_CONTRACT

        objective = "grammar_source_pairs" if schema.endswith(".v4") else "grammar_choices"
        if (plan.get("objective") != objective
                or plan.get("loss_scope") != "semantic_decisions"
                or plan.get("grammar_choice_contract") != GRAMMAR_CHOICE_CONTRACT
                or (schema.endswith(".v4") and (
                    plan.get("grammar_source_pair_contract") != SOURCE_PAIR_CONTRACT
                    or not plan.get("grammar_source_pair_fit_partners")))):
            raise ValueError("native fit grammar-choice contract differs")
        if schema.endswith(".v4") and "source_evidence_control" in plan:
            raise ValueError("native paired-source fit cannot erase its training source")
        if "source_evidence_control" not in plan:
            return "source_text"
        if plan["source_evidence_control"] == SOURCE_ERASURE_CONTRACT:
            return "source_token_erasure"
        raise ValueError("native fit source-evidence control contract differs")
    if plan.get("objective") == "grammar_choices" or "grammar_choice_contract" in plan:
        raise ValueError("historical native fit cannot acquire grammar-choice supervision")
    if schema == "aura.semantic_native_fit_plan.v1" and "source_evidence_control" not in plan:
        return "source_text"
    if (schema == "aura.semantic_native_fit_plan.v2"
            and plan.get("source_evidence_control") == SOURCE_ERASURE_CONTRACT):
        return "source_token_erasure"
    raise ValueError("native fit source-evidence control contract differs")


def erase_native_source_tokens(sequence: NativeProgramSequence, source: str, tokenizer: Any
                               ) -> tuple[NativeProgramSequence, dict[str, Any]]:
    """A training/lesion control, retaining length but no source-content tokens."""
    from core.learning.semantic_program_feature_materialization import (
        offset_tokenizer_for_worker,
        tokenize_with_offsets,
    )

    if (not isinstance(sequence, NativeProgramSequence) or not isinstance(source, str) or not source
            or type(sequence.continuation_start) is not int
            or not 0 < sequence.continuation_start < len(sequence.tokens)
            or any(type(index) is not int or not sequence.continuation_start <= index < len(sequence.tokens)
                   for index in sequence.semantic_positions)):
        raise ValueError("native source control needs a complete source-bound sequence")
    marker = "_AURA_NATIVE_SOURCE_CONTENT_ANCHOR_"
    anchored = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], add_generation_prompt=True, tokenize=False)
    prefix = tokenizer.apply_chat_template(
        [{"role": "user", "content": source}], add_generation_prompt=True, tokenize=False)
    if not isinstance(anchored, str) or anchored.count(marker) != 1 or not isinstance(prefix, str):
        raise ValueError("native source control cannot locate the user-content boundary")
    before, after = anchored.split(marker)
    if prefix != before + source + after:
        raise ValueError("native source control template transforms user content")
    start, end = len(before), len(before) + len(source)
    prefix_tokens, offsets = tokenize_with_offsets(offset_tokenizer_for_worker(tokenizer), prefix)
    positions = tuple(index for index, (left, right) in enumerate(offsets)
                      if left < end and right > start)
    if (not positions or positions != tuple(range(positions[0], positions[-1] + 1))
            or offsets[positions[0]][0] != start or offsets[positions[-1]][1] != end
            or any(not start <= offsets[index][0] < offsets[index][1] <= end
                   or index >= sequence.continuation_start
                   or prefix_tokens[index] != sequence.tokens[index] for index in positions)):
        raise ValueError("native source control cannot erase content without changing its boundary")
    filler = tokenizer.encode("?", add_special_tokens=False)
    if len(filler) != 1 or tokenizer.decode(filler, skip_special_tokens=False) != "?":
        raise ValueError("native source control filler is not one literal token")
    tokens = list(sequence.tokens)
    for index in positions:
        tokens[index] = filler[0]
    controlled = replace(sequence, tokens=tuple(tokens))
    erased = set(positions)
    if (controlled.tokens[sequence.continuation_start:] != sequence.tokens[sequence.continuation_start:]
            or any(controlled.tokens[index] != token for index, token in enumerate(sequence.tokens)
                   if index not in erased)):
        raise ValueError("native source control changed template or supervision tokens")
    receipt = {
        "schema": "aura.semantic_native_source_erasure.v1",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "sequence_tokens": len(sequence.tokens),
        "erased_source_positions": positions,
        "erased_source_tokens": len(positions),
        "filler_token_id": filler[0],
        "continuation_start": sequence.continuation_start,
        "semantic_positions": sequence.semantic_positions,
        "template_tokens_unchanged": True,
        "supervision_tokens_unchanged": True,
        "sequence_length_unchanged": True,
        "retained_nuisances": ("source_token_length", "template_position", "native_output_prefix"),
        "source_content_tokens_available": False,
        "serving_authority": False,
    }
    return controlled, receipt
