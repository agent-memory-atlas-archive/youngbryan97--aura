"""Train the same grammar competitions without averaging away weak bindings."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import mlx.core as mx


GRAMMAR_PATH_CONTRACT = {
    "schema": "aura.native_grammar_path_objective.v1",
    "aggregation": "logmeanexp_conditional_decision_losses",
    "decision_inventory": "all_teacher_path_competitions_and_declared_source_pair_interaction",
    "gradient_weights": "softmax_of_conditional_decision_losses",
    "equal_losses_match_mean_gradient": True,
    "calibration": "same_source_only_path_objective",
    "held_labels_used": False,
    "unseen_correctness_guaranteed": False,
}

JOINT_GRAPH_CONTRAST_CONTRACT = {
    "schema": "aura.native_joint_graph_contrast.v1",
    "positive": "source_training_target_complete_graph",
    "negative": "typed_source_only_counterfactually_witnessed_complete_graph",
    "score": "summed_native_semantic_decision_log_probability",
    "training": "one_whole_graph_loss_in_same_path_risk",
    "calibration": "same_complete_graph_competition_with_source_only_targets",
    "inference": "search_complete_candidates_then_score_whole_graph",
    "held_labels_used": False,
    "serving_authority": False,
}


def path_choice_contract() -> dict:
    from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT

    return {**GRAMMAR_CHOICE_CONTRACT,
            "loss": "logmeanexp_conditional_decision_and_source_interaction_losses"}


def native_path_risk_loss(losses: mx.array) -> mx.array:
    """Smooth worst-choice risk, with the mean's scale on equally hard choices.

    mean(losses) <= log(mean(exp(losses))) <= max(losses). Its gradient
    allocates more weight to a weak choice, regardless of operation kind.
    It is a surrogate, not a certificate that an unseen source is correct.
    """
    import mlx.core as mx

    if losses.ndim != 1 or losses.shape[0] < 1:
        raise ValueError("native path objective needs every conditional decision loss")
    return mx.logsumexp(losses) - math.log(losses.shape[0])


def native_grammar_path_objective(scorer: Callable[[str], mx.array], decisions: Sequence[Any], *,
                                  partner_decisions: Sequence[Any] | None = None,
                                  pair: Mapping[str, Any] | None = None,
                                  typed_pairs: Sequence[Mapping[str, Any]] | None = None,
                                  partners: Mapping[str, Sequence[Any]] | None = None,
                                  measured_scores: list[mx.array] | None = None,
                                  graph_keys: tuple[str, ...] | None = None,
                                  measured_graph_scores: list[mx.array] | None = None) -> mx.array:
    """Compete all weak choices and the source contrast on one loss scale.

    An operation-only source interaction must not receive the weight of an
    entire path while each reference receives only a fraction of that path.
    Existing alternatives and source-positive labels remain unchanged.
    """
    import mlx.core as mx

    from core.learning.semantic_native_decision_supervision import native_decision_choice_loss
    from core.learning.semantic_native_program import native_choice_loss
    from core.learning.semantic_native_source_pairs import native_source_interaction_loss

    if (not decisions or (pair is None) != (partner_decisions is None)
            or (typed_pairs is None) != (partners is None)
            or (pair is not None and typed_pairs is not None)
            or graph_keys is not None and (not isinstance(graph_keys, tuple)
                or len(graph_keys) < 2 or len(set(graph_keys)) != len(graph_keys))
            or measured_graph_scores is not None and graph_keys is None):
        raise ValueError("native path objective source/partner decisions differ")
    losses, scores = [], []
    for keys, correct in decisions:
        values = mx.stack([scorer(key) for key in keys])
        scores.append(values)
        losses.append(native_decision_choice_loss(values, correct))
    contrasts = [] if pair is None else [(pair, partner_decisions)]
    if typed_pairs is not None:
        kinds = [row["kind"] for row in typed_pairs]
        if (not typed_pairs or len(kinds) != len(set(kinds))
                or not set(kinds) <= {"operation", "reference", "termination"}
                or any(row["partner"] not in partners for row in typed_pairs)):
            raise ValueError("native path objective typed contrast inventory differs")
        contrasts = [(row, partners[row["partner"]]) for row in typed_pairs]
    for contrast, peer_decisions in contrasts:
        ordinal, own, rival = (contrast[key] for key in ("decision_index", "own_index", "partner_index"))
        if (type(ordinal) is not int or not 0 <= ordinal < min(len(decisions), len(peer_decisions))
                or type(own) is not int or type(rival) is not int or own == rival
                or not 0 <= min(own, rival) <= max(own, rival) < scores[ordinal].shape[0]
                or len(peer_decisions[ordinal][0]) != scores[ordinal].shape[0]
                or decisions[ordinal][1] != own or peer_decisions[ordinal][1] != rival):
            raise ValueError("native path objective contrast labels differ")
        partner_keys = peer_decisions[ordinal][0]
        losses.append(native_source_interaction_loss(scores[ordinal][own], scores[ordinal][rival],
            scorer(partner_keys[own]), scorer(partner_keys[rival])))
    if graph_keys is not None:
        graph_scores = mx.stack([scorer(key) for key in graph_keys])
        losses.append(native_choice_loss(graph_scores, (0,)))
        if measured_graph_scores is not None:
            measured_graph_scores.append(graph_scores)
    if measured_scores is not None:
        measured_scores.extend(scores)
    return native_path_risk_loss(mx.stack(losses))
