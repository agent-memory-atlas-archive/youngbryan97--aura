"""Share full-shape inference only where every scored causal prefix agrees."""

from __future__ import annotations

from dataclasses import dataclass

from core.learning.semantic_native_program import NativeProgramSequence
from core.verify.invariants import invariant


@dataclass(frozen=True)
class NativeCausalGroup:
    representative: int
    members: tuple[int, ...]
    prediction_positions: tuple[int, ...]


def native_causal_groups(sequences: tuple[NativeProgramSequence, ...]) -> tuple[NativeCausalGroup, ...]:
    """Group equal shapes and prefixes through the last supervised predecessor.

    A representative keeps its entire original input, including future tokens.
    Other members differ only after all selected prediction states. No padding,
    batching, truncation, cache advancement, or changed token alignment occurs.
    This is an inference optimization, not permission to change training.
    """
    if not isinstance(sequences, tuple) or not 1 <= len(sequences) <= 1024:
        raise ValueError("native causal grouping needs a bounded sequence tuple")
    grouped = {}
    for index, sequence in enumerate(sequences):
        tokens, positions, boundary = sequence.tokens, sequence.semantic_positions, sequence.continuation_start
        if (not isinstance(tokens, tuple) or len(tokens) < 2
                or any(type(token) is not int or token < 0 for token in tokens)
                or type(boundary) is not int or not 1 <= boundary < len(tokens)
                or not isinstance(positions, tuple) or not positions
                or any(type(position) is not int or not boundary <= position < len(tokens)
                       for position in positions)
                or tuple(sorted(set(positions))) != positions):
            raise ValueError("native causal grouping lost its supervised token boundary")
        # Target position p is predicted by hidden state p-1. Include every
        # input through that predecessor, not the target at p.
        key = (len(tokens), boundary, positions, tokens[:positions[-1]])
        grouped.setdefault(key, []).append(index)
    return tuple(NativeCausalGroup(members[0], tuple(members),
                                  tuple(position - 1 for position in key[2]))
                 for key, members in grouped.items())


def score_native_causal_groups(prefix, suffix, sequences: tuple[NativeProgramSequence, ...],
                               *, branches=None):
    """Score each original target with shared full-shape causal logits.

    Callers must qualify this path on the real decoder before using it in a
    measured candidate. The result grants no serving or selection authority.
    Sharing applies within this call only; parameter changes cannot reuse it.
    A frozen branch owner may also reuse the source prefix across calls.
    """
    import mlx.core as mx
    groups = native_causal_groups(sequences)
    if any(child.training for module in (*suffix.layers, suffix.norm, suffix.output)
           for child in module.modules()):
        raise ValueError("native causal sharing requires suffix evaluation mode")
    if branches is None:
        states = (prefix.capture(mx.array([sequences[group.representative].tokens[:-1]],
                                          dtype=mx.int32)) for group in groups)
    else:
        states = branches.capture_many(tuple(sequences[group.representative].tokens[:-1]
                                             for group in groups))
    scores = [None] * len(sequences)
    for group, hidden in zip(groups, states, strict=True):
        logits = suffix(hidden, logit_positions=group.prediction_positions).astype(mx.float32)
        if logits.shape[:2] != (1, len(group.prediction_positions)):
            raise ValueError("native causal sharing lost its selected logit alignment")
        log_norm = mx.logsumexp(logits, axis=-1)
        for index in group.members:
            sequence = sequences[index]
            targets = mx.array([[sequence.tokens[position] for position in sequence.semantic_positions]],
                               dtype=mx.int32)
            target_logits = mx.take_along_axis(logits, targets[..., None], axis=-1).squeeze(-1)
            scores[index] = -mx.sum(log_norm - target_logits).item()
    return tuple(scores), {
        "schema": "aura.native_causal_group_execution.v1",
        "alternatives": len(sequences), "full_single_row_forwards": len(groups),
        "vocabulary_normalizations": len(groups),
        "shared_forwards": len(sequences) - len(groups),
        "groups": [{"representative": group.representative, "members": list(group.members),
                    "prediction_positions": list(group.prediction_positions)} for group in groups],
        "numeric_equivalence_measured": False, "serving_authority": False,
    }


@invariant("learning.native_causal_groups_preserve_prediction_prefixes", scope="learning",
           owner="core/learning/semantic_native_causal_groups.py", observational=False)
def _causal_group_boundary() -> tuple:
    sequences = (NativeProgramSequence((1, 2, 3, 4, 5), 2, semantic_positions=(3,)),
                 NativeProgramSequence((1, 2, 3, 8, 9), 2, semantic_positions=(3,)),
                 NativeProgramSequence((1, 2, 7, 4, 5), 2, semantic_positions=(3,)))
    groups = native_causal_groups(sequences)
    return ((groups[0].members == (0, 1) and groups[1].members == (2,)),
            {"groups": len(groups), "causal_prefix_changed_in_separate_group": True})
