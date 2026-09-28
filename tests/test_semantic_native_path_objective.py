"""Worst-choice risk changes gradients, not labels, grammar, or answer authority."""

import math

import mlx.core as mx
import pytest

from core.learning.semantic_native_path_objective import (
    native_grammar_path_objective,
    native_path_risk_loss,
)


def test_risk_bounds_and_stable_large_losses():
    for values in ((.01, .02, 3.), (1000., 1001., 1002.), (0.,)):
        losses = mx.array(values, dtype=mx.float32)
        risk = native_path_risk_loss(losses).item()
        assert sum(values) / len(values) - 1e-4 <= risk <= max(values) + 1e-4


def test_equal_difficulty_preserves_mean_loss_and_gradient_scale():
    losses = mx.array([2., 2., 2., 2.])
    assert native_path_risk_loss(losses).item() == pytest.approx(2.)
    assert mx.grad(native_path_risk_loss)(losses).tolist() == pytest.approx([.25] * 4)


def test_hardest_binding_receives_more_gradient_without_kind_or_family_features():
    losses = mx.array([.01, 2., .01, .01])
    gradient = mx.grad(native_path_risk_loss)(losses).tolist()
    assert gradient[1] > .25
    assert sum(gradient) == pytest.approx(1.)
    expected = math.exp(2.) / (math.exp(2.) + 3 * math.exp(.01))
    assert gradient[1] == pytest.approx(expected)


def test_many_deterministic_choices_do_not_hide_a_hard_decision_gradient():
    losses = mx.array([0.] * 15 + [3.])
    gradient = mx.grad(native_path_risk_loss)(losses).tolist()
    assert gradient[-1] > 1 / 16
    assert gradient[-1] > .5


@pytest.mark.parametrize("shape", [(0,), (1, 2)])
def test_empty_or_nonpath_losses_are_rejected(shape):
    with pytest.raises(ValueError):
        native_path_risk_loss(mx.zeros(shape))


def test_pair_and_binding_are_one_competition_not_additive_operation_dominance():
    decisions = ((('op-correct', 'op-rival'), 0), (('ref-correct', 'ref-rival'), 0))
    partner = ((('partner-op', 'partner-rival'), 1),)
    pair = {"decision_index": 0, "own_index": 0, "partner_index": 1}
    fixed = {'op-correct': 3., 'op-rival': 0., 'partner-op': 0., 'partner-rival': 3., 'ref-rival': 2.}

    def objective(reference):
        return native_grammar_path_objective(lambda key: reference if key == 'ref-correct'
            else mx.array(fixed[key]), decisions, pair=pair, partner_decisions=partner)

    derivative = mx.grad(objective)(mx.array(0.)).item()
    assert derivative < -.5
    assert objective(mx.array(3.)).item() < objective(mx.array(0.)).item()


def test_calibration_collects_same_scores_without_partner_or_runtime_labels():
    scores = []
    loss = native_grammar_path_objective(lambda key: mx.array(float(key)),
        (((1, 2), 1), ((3,), 0)), measured_scores=scores)
    assert len(scores) == 2
    assert scores[0].tolist() == [1., 2.]
    assert loss.item() >= 0.


def test_complete_graph_contrast_shares_the_path_risk_and_changes_gradient():
    decisions = ((("op-correct", "op-rival"), 0),)
    fixed = {"op-correct": mx.array(2.), "op-rival": mx.array(0.),
             "graph-rival": mx.array(2.)}
    measured = []

    def objective(graph):
        return native_grammar_path_objective(
            lambda key: graph if key == "graph-correct" else fixed[key], decisions,
            graph_keys=("graph-correct", "graph-rival"),
            measured_graph_scores=measured)

    assert mx.grad(objective)(mx.array(0.)).item() < 0.
    assert objective(mx.array(4.)).item() < objective(mx.array(0.)).item()
    assert len(measured) == 3
    assert measured[0].tolist() == pytest.approx([0., 2.])
    with pytest.raises(ValueError, match="source/partner"):
        native_grammar_path_objective(lambda key: fixed[key], decisions,
                                      graph_keys=("same", "same"))


def test_mismatched_pair_cannot_silently_move_loss_to_another_target():
    with pytest.raises(ValueError, match="contrast labels"):
        native_grammar_path_objective(lambda key: mx.array(float(key)),
            (((1, 2), 0),), partner_decisions=(((3, 4), 0),),
            pair={"decision_index": 0, "own_index": 0, "partner_index": 1})


def test_typed_source_interactions_share_the_path_risk_and_train_reference_binding():
    from core.learning.semantic_native_decision_supervision import native_decision_choice_loss
    from core.learning.semantic_native_source_pairs import native_source_interaction_loss

    decisions = ((('op', 'op-rival'), 0), (('ref', 'ref-rival'), 0))
    peers = {"op-source": ((('peer-op', 'peer-op-rival'), 1),),
             "ref-source": ((('unused', 'unused2'), 0), (('peer-ref', 'peer-ref-rival'), 1))}
    pairs = [{"kind": kind, "partner": partner, "decision_index": index,
              "own_index": 0, "partner_index": 1}
             for index, (kind, partner) in enumerate([
                 ("operation", "op-source"), ("reference", "ref-source")])]
    fixed = {"op": 3., "op-rival": 0., "peer-op": 0., "peer-op-rival": 3.,
             "ref-rival": 2., "peer-ref": 2., "peer-ref-rival": 0.}

    def objective(reference):
        return native_grammar_path_objective(lambda key: reference if key == "ref"
            else mx.array(fixed[key]), decisions, typed_pairs=pairs, partners=peers)

    with mx.stream(mx.cpu):
        expected = native_path_risk_loss(mx.stack([
            native_decision_choice_loss(mx.array([3., 0.]), 0),
            native_decision_choice_loss(mx.array([0., 2.]), 0),
            native_source_interaction_loss(*[mx.array(value) for value in [3., 0., 0., 3.]]),
            native_source_interaction_loss(*[mx.array(value) for value in [0., 2., 2., 0.]]),
        ]))
        assert objective(mx.array(0.)).item() == pytest.approx(expected.item())
        assert mx.grad(objective)(mx.array(0.)).item() < -.5
        assert objective(mx.array(4.)).item() < objective(mx.array(0.)).item()


@pytest.mark.parametrize("defect", ["duplicate", "unknown_kind", "missing_peer", "labels", "legacy_mix"])
def test_typed_interaction_rejects_bad_inventory_or_rebound_target(defect):
    decisions = (((1, 2), 0),)
    pairs = [{"kind": "reference", "partner": "peer", "decision_index": 0,
              "own_index": 0, "partner_index": 1}]
    peers = {"peer": (((3, 4), 1),)}
    if defect == "duplicate":
        pairs *= 2
    elif defect == "unknown_kind":
        pairs[0]["kind"] = "family"
    elif defect == "missing_peer":
        peers.clear()
    elif defect == "labels":
        peers["peer"] = (((3, 4), 0),)
    extra = {"pair": pairs[0], "partner_decisions": peers["peer"]} if defect == "legacy_mix" else {}
    with pytest.raises(ValueError, match="source/partner|typed contrast|contrast labels"):
        native_grammar_path_objective(lambda key: mx.array(float(key)), decisions,
                                     typed_pairs=pairs, partners=peers, **extra)
