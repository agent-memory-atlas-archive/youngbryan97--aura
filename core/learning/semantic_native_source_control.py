"""Erase source content without changing native template or target positions."""

from __future__ import annotations

import hashlib
import json
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
    graph_mode = schema == "aura.semantic_native_fit_plan.v7"
    typed_mode = schema == "aura.semantic_native_fit_plan.v6" or (
        graph_mode and "grammar_source_pair_inventory" in plan)
    path_mode = schema in {"aura.semantic_native_fit_plan.v5", "aura.semantic_native_fit_plan.v6",
                           "aura.semantic_native_fit_plan.v7"}
    graph_fields = {"joint_graph_contrast_limit", "graph_contrast_contract"}
    if not graph_mode and graph_fields & set(plan):
        raise ValueError("historical native fit cannot acquire whole-graph training")
    if graph_mode:
        from core.learning.semantic_native_path_objective import JOINT_GRAPH_CONTRAST_CONTRACT

        if (plan.get("graph_contrast_contract") != JOINT_GRAPH_CONTRAST_CONTRACT
                or type(plan.get("joint_graph_contrast_limit")) is not int
                or not 2 <= plan["joint_graph_contrast_limit"] <= 32
                or plan.get("prefix_storage_contract", {}).get("mode") != "source_shards"
                or plan.get("execution_contract", {}).get("prefix_strategy") != "trie"
                or "reused_prefix_contract" in plan):
            raise ValueError("native whole-graph contrast contract differs")
    if not typed_mode and "grammar_source_pair_inventory" in plan:
        raise ValueError("historical native fit cannot acquire typed source-pair coverage")
    path_fields = {"grammar_path_objective_contract", "path_checkpoint_selection_contract"}
    if not path_mode and path_fields & set(plan):
        raise ValueError("historical native fit cannot acquire path-risk selection")
    if schema in {"aura.semantic_native_fit_plan.v3", "aura.semantic_native_fit_plan.v4",
                  "aura.semantic_native_fit_plan.v5", "aura.semantic_native_fit_plan.v6",
                  "aura.semantic_native_fit_plan.v7"}:
        from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
        from core.learning.semantic_native_source_pairs import SOURCE_PAIR_CONTRACT

        choice_contract = GRAMMAR_CHOICE_CONTRACT
        objective = "grammar_source_pairs" if schema.endswith(".v4") else "grammar_choices"
        if path_mode:
            from core.learning.semantic_native_path_objective import (
                GRAMMAR_PATH_CONTRACT,
                path_choice_contract,
            )
            from core.learning.semantic_native_path_selection import (
                JOINT_GRAPH_SELECTION_CONTRACT,
                PATH_SELECTION_CONTRACT,
            )

            objective = plan.get("objective")
            choice_contract = path_choice_contract()
            if (objective not in {"grammar_choices", "grammar_source_pairs"}
                    or plan.get("grammar_path_objective_contract") != GRAMMAR_PATH_CONTRACT
                    or plan.get("path_checkpoint_selection_contract") != (
                        JOINT_GRAPH_SELECTION_CONTRACT if graph_mode else PATH_SELECTION_CONTRACT)
                    or plan.get("selection") != (
                        "baseline_preserving_joint_source_calibration" if graph_mode else
                        "baseline_preserving_complete_source_calibration_paths")
                    or plan.get("unfitted_checkpoint_eligible") is not True):
                raise ValueError("native path-risk objective or selection contract differs")
        paired = objective == "grammar_source_pairs"
        pair_contract = SOURCE_PAIR_CONTRACT
        if typed_mode:
            from core.learning.semantic_native_typed_source_pairs import (
                TYPED_SOURCE_PAIR_CONTRACT,
                typed_source_pair_inventory,
            )

            pairs = plan.get("grammar_source_pair_fit_partners", {})
            fit_ids = set(plan.get("fit_ids", ()))
            schedule = plan.get("scheduled_fit_ids", ())
            if (not paired or not pairs or not set(pairs) <= fit_ids
                    or not set(schedule) <= fit_ids
                    or any(not isinstance(rows, list) or not rows
                           or any(not isinstance(row, dict) or row.get("partner") not in fit_ids
                                  for row in rows)
                           for rows in pairs.values())
                    or plan.get("grammar_source_pair_inventory") != typed_source_pair_inventory(
                        pairs, schedule)
                    or plan.get("grammar_source_pair_updates")
                        != plan["grammar_source_pair_inventory"]["paired_updates"]):
                raise ValueError("native typed source-pair coverage contract differs")
            pair_contract = TYPED_SOURCE_PAIR_CONTRACT
        if (plan.get("objective") != objective
                or plan.get("loss_scope") != "semantic_decisions"
                or plan.get("grammar_choice_contract") != choice_contract
                or (paired and (
                    plan.get("grammar_source_pair_contract") != pair_contract
                    or not plan.get("grammar_source_pair_fit_partners")))):
            raise ValueError("native fit grammar-choice contract differs")
        if (paired or path_mode) and "source_evidence_control" in plan:
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


def apply_native_source_evidence(sequence: NativeProgramSequence, source: str,
                                 tokenizer, *, mode: str):
    """Apply the same source-content intervention to any native scoring path."""
    if mode == "source_text":
        return sequence, None
    if mode == "source_token_erasure":
        return erase_native_source_tokens(sequence, source, tokenizer)
    raise ValueError("unknown native source-evidence mode")


def native_score_input_receipt(sequence: NativeProgramSequence,
                               control: dict[str, Any] | None) -> dict[str, Any]:
    """Bind one scored choice to its exact tokens and source intervention."""
    if not isinstance(sequence, NativeProgramSequence):
        raise ValueError("native score input needs a native sequence")
    if control is not None and (not isinstance(control, dict)
                                or control.get("source_content_tokens_available") is not False
                                or type(control.get("erased_source_tokens")) is not int
                                or control["erased_source_tokens"] < 1):
        raise ValueError("native score input erasure receipt is invalid")

    def sha(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"))
                              .encode()).hexdigest()

    return {
        "schema": "aura.native_score_input.v1",
        "sequence_sha256": sha({"tokens": sequence.tokens,
                                "continuation_start": sequence.continuation_start,
                                "semantic_positions": sequence.semantic_positions}),
        "source_control_sha256": None if control is None else sha(control),
        "erased_source_tokens": 0 if control is None else control["erased_source_tokens"],
    }
