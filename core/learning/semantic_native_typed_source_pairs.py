"""Give every witnessed grammar decision kind its own fit-local source contrast."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

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
    "inventory": "one_witnessed_partner_per_available_conditional_decision_per_source",
    "selection": "prefer_shared_lineage_then_token_multiset_overlap_then_source_sha256",
    "loss": "logistic_source_by_choice_interaction",
    "aggregation": "joint_logmeanexp_with_all_conditional_path_losses",
    "calibration": "ordinary_source_only_conditional_path_risk",
    "held_labels_used": False,
    "construction_labels_are_model_features": False,
    "unseen_correctness_guaranteed": False,
}

_KINDS = ("operation", "reference", "termination")


def _shared_divergence(left: Sequence[Any], right: Sequence[Any]) -> dict | None:
    for ordinal, (own, peer) in enumerate(zip(left, right, strict=False)):
        if (own.kind != peer.kind or tuple(choice.value for choice in own.choices)
                != tuple(choice.value for choice in peer.choices)):
            return None
        if own.correct_index != peer.correct_index:
            return {"decision_index": ordinal, "kind": own.kind,
                    "own_index": own.correct_index, "partner_index": peer.correct_index}
    return None


def _decision_signature(decision: Any) -> tuple:
    return decision.kind, tuple(choice.value for choice in decision.choices)


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
    token_counts = {identity: Counter(item.ir.source_token_ids) for identity, item in by_id.items()}
    token_lengths = {identity: sum(counts.values()) for identity, counts in token_counts.items()}
    decisions = {identity: native_teacher_decisions(programs[identity], types[identity],
                    register_encoding=register_encoding) for identity in by_id}
    prefix_groups: dict[tuple, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    source_keys: dict[str, list[tuple]] = {}
    for identity, rows in decisions.items():
        prefix: list[tuple] = []
        keys = []
        for row in rows:
            signature = _decision_signature(row)
            key = (types[identity], tuple(prefix), signature)
            prefix_groups[key][row.correct_index].append(identity)
            keys.append(key)
            prefix.append((*signature, row.correct_index))
        source_keys[identity] = keys
    pairs = {}
    for identity, item in sorted(by_id.items()):
        selected = []
        for decision_index, key in enumerate(source_keys[identity]):
            alternatives = prefix_groups[key]
            own_index = decisions[identity][decision_index].correct_index
            other_ids = [peer for choice, members in alternatives.items()
                         if choice != own_index for peer in members]
            for shared_lineage in (True, False):
                ranked = []
                for partner_id in other_ids:
                    partner = by_id[partner_id]
                    same = bool(item.contrast_id and item.contrast_id == partner.contrast_id)
                    if same != shared_lineage:
                        continue
                    overlap = (2 * sum((token_counts[identity] & token_counts[partner_id]).values())
                               / (token_lengths[identity] + token_lengths[partner_id]))
                    ranked.append((-overlap, partner_id))
                for _overlap, partner_id in sorted(ranked):
                    divergence = _shared_divergence(decisions[identity], decisions[partner_id])
                    if divergence is None or divergence["decision_index"] != decision_index:
                        raise ValueError("typed source prefix index disagrees with grammar")
                    comparison = compare_program_meanings(programs[identity], programs[partner_id],
                        counterfactual_inputs(item.public_inputs))
                    if comparison["status"] != "different" or comparison.get("witness") is None:
                        continue
                    witness = comparison["witness"]
                    selected.append({"partner": partner_id, **divergence, "witness": witness,
                        "witness_sha256": hashlib.sha256(json.dumps(witness, sort_keys=True,
                            allow_nan=False).encode()).hexdigest()})
                    break
                else:
                    continue
                break
        if selected:
            pairs[identity] = selected
    return pairs


def typed_source_pair_inventory(pairs: dict, schedule: Iterable[str]) -> dict:
    """Count observed coverage without declaring absent decision kinds present."""
    scheduled = Counter(schedule)
    source_counts, interaction_counts = Counter(), Counter()
    decision_sources, decision_updates = Counter(), Counter()
    for identity, rows in pairs.items():
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("typed source contrast inventory needs declared pair rows")
        kinds = [row.get("kind") for row in rows]
        indices = [row.get("decision_index") for row in rows]
        if (not rows or not set(kinds) <= set(_KINDS)
                or any(type(index) is not int or index < 0 for index in indices)
                or indices != sorted(set(indices))):
            raise ValueError("typed source contrast inventory repeats or invents a decision")
        if identity not in scheduled:
            raise ValueError("typed source contrast inventory contains an unscheduled source")
        source_counts.update(set(kinds))
        for kind in kinds:
            interaction_counts[kind] += scheduled[identity]
        decision_sources.update(indices)
        decision_updates.update({index: scheduled[identity] for index in indices})
    return {"paired_sources": len(pairs),
            "paired_updates": sum(scheduled[identity] for identity in pairs),
            "sources_by_kind": {kind: source_counts[kind] for kind in _KINDS},
            "interactions_by_kind": {kind: interaction_counts[kind] for kind in _KINDS},
            "sources_by_decision": {str(index): count for index, count in sorted(decision_sources.items())},
            "interactions_by_decision": {str(index): count for index, count in sorted(decision_updates.items())},
            "held_labels_used": False}


@invariant("learning.native_typed_contrasts_do_not_invent_coverage", scope="learning",
           owner="core/learning/semantic_native_typed_source_pairs.py", observational=False)
def _typed_contrast_coverage() -> dict:
    rows = {"source": [{"kind": "reference", "partner": "peer", "decision_index": 1}]}
    measured = typed_source_pair_inventory(rows, ("source", "source", "peer"))
    assert measured["sources_by_kind"] == {"operation": 0, "reference": 1, "termination": 0}
    assert measured["interactions_by_kind"]["reference"] == 2
    try:
        typed_source_pair_inventory({"source": rows["source"] * 2}, ("source", "peer"))
    except ValueError:
        return measured
    raise AssertionError("duplicate typed source contrasts acquired training weight")
