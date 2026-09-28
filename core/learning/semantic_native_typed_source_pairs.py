"""Give every witnessed grammar decision kind its own fit-local source contrast."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from core.learning.semantic_graph_counterexamples import (
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_native_decision_supervision import native_teacher_decisions
from core.learning.semantic_native_source_pairs import _input_types
from core.verify.invariants import invariant

if TYPE_CHECKING:
    from core.learning.semantic_program_transducer import SemanticTransducerTrainingExample


TYPED_SOURCE_PAIR_CONTRACT = {
    "basis": "complete_fit_partition_shared_typed_prefix_and_witnessed_meaning_change",
    "decision": "first_shared_typed_grammar_choice_with_opposite_targets",
    "inventory": "one_witnessed_partner_per_available_decision_kind_per_source",
    "selection": "prefer_shared_lineage_then_max_token_overlap_then_source_sha256",
    "loss": "logistic_source_by_choice_interaction",
    "aggregation": "joint_logmeanexp_with_all_conditional_path_losses",
    "calibration": "ordinary_source_only_conditional_path_risk",
    "held_labels_used": False,
    "construction_labels_are_model_features": False,
    "unseen_correctness_guaranteed": False,
}

_KINDS = ("operation", "reference", "termination")


def _shared_divergence(left, right):
    for ordinal, (own, peer) in enumerate(zip(left, right, strict=False)):
        if (own.kind != peer.kind or tuple(choice.value for choice in own.choices)
                != tuple(choice.value for choice in peer.choices)):
            return None
        if own.correct_index != peer.correct_index:
            return {"decision_index": ordinal, "kind": own.kind,
                    "own_index": own.correct_index, "partner_index": peer.correct_index}
    return None


def native_typed_source_pair_plan(examples: Iterable[SemanticTransducerTrainingExample],
                                  fit_ids: Iterable[str], *, register_encoding: str) -> dict:
    """Select by fit-only supervision; retain an execution witness for every pair.

    A different lineage can supply a reference contrast when the same lineage
    supplies only operations. Peer ranking never enters the model's features.
    Neither calibration nor held sources may supply a partner or a witness.
    """
    fit_ids = tuple(fit_ids)
    requested = set(fit_ids)
    fitting = [item for item in examples if item.ir.source_text_sha256 in requested]
    by_id = {item.ir.source_text_sha256: item for item in fitting}
    if (not requested or len(requested) != len(fit_ids) or len(by_id) != len(fitting)
            or set(by_id) != requested or any(item.split != "train" for item in fitting)):
        raise ValueError("typed source pairs require the unique complete fit-only partition")
    programs = {identity: item.ir.to_program() for identity, item in by_id.items()}
    types = {identity: _input_types(item.public_inputs) for identity, item in by_id.items()}
    decisions = {identity: native_teacher_decisions(programs[identity], types[identity],
                    register_encoding=register_encoding) for identity in by_id}
    pairs = {}
    for identity, item in sorted(by_id.items()):
        options = {kind: [] for kind in _KINDS}
        for partner_id, partner in by_id.items():
            if partner_id == identity or types[identity] != types[partner_id]:
                continue
            divergence = _shared_divergence(decisions[identity], decisions[partner_id])
            if divergence is None:
                continue
            overlap = SequenceMatcher(None, tuple(item.ir.source_token_ids),
                tuple(partner.ir.source_token_ids), autojunk=False).ratio()
            shared_lineage = bool(item.contrast_id and item.contrast_id == partner.contrast_id)
            options[divergence["kind"]].append((not shared_lineage, -overlap,
                                                 partner_id, divergence))
        selected = []
        for kind in _KINDS:
            for _other_lineage, _overlap, partner_id, divergence in sorted(options[kind]):
                comparison = compare_program_meanings(programs[identity], programs[partner_id],
                    counterfactual_inputs(item.public_inputs))
                if comparison["status"] != "different" or comparison.get("witness") is None:
                    continue
                witness = comparison["witness"]
                selected.append({"partner": partner_id, **divergence, "witness": witness,
                    "witness_sha256": hashlib.sha256(json.dumps(witness, sort_keys=True,
                        allow_nan=False).encode()).hexdigest()})
                break
        if selected:
            pairs[identity] = selected
    return pairs


def typed_source_pair_inventory(pairs: dict, schedule: Iterable[str]) -> dict:
    """Count observed coverage without declaring absent decision kinds present."""
    scheduled = Counter(schedule)
    source_counts, interaction_counts = Counter(), Counter()
    for identity, rows in pairs.items():
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("typed source contrast inventory needs declared pair rows")
        kinds = [row.get("kind") for row in rows]
        if not rows or len(kinds) != len(set(kinds)) or not set(kinds) <= set(_KINDS):
            raise ValueError("typed source contrast inventory repeats or invents a decision kind")
        if identity not in scheduled:
            raise ValueError("typed source contrast inventory contains an unscheduled source")
        source_counts.update(kinds)
        interaction_counts.update({kind: scheduled[identity] for kind in kinds})
    return {"paired_sources": len(pairs),
            "paired_updates": sum(scheduled[identity] for identity in pairs),
            "sources_by_kind": {kind: source_counts[kind] for kind in _KINDS},
            "interactions_by_kind": {kind: interaction_counts[kind] for kind in _KINDS},
            "held_labels_used": False}


@invariant("learning.native_typed_contrasts_do_not_invent_coverage", scope="learning",
           owner="core/learning/semantic_native_typed_source_pairs.py", observational=False)
def _typed_contrast_coverage() -> dict:
    rows = {"source": [{"kind": "reference", "partner": "peer"}]}
    measured = typed_source_pair_inventory(rows, ("source", "source", "peer"))
    assert measured["sources_by_kind"] == {"operation": 0, "reference": 1, "termination": 0}
    assert measured["interactions_by_kind"]["reference"] == 2
    try:
        typed_source_pair_inventory({"source": rows["source"] * 2}, ("source", "peer"))
    except ValueError:
        return measured
    raise AssertionError("duplicate typed source contrasts acquired training weight")
