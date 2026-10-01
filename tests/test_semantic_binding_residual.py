"""Nonlinear source binding reaches the existing proposer and grounded solver."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_binding_residual import NonlinearBindingResidual, fit_binding_residual
from core.learning.semantic_triadic_binding import TriadicBindingHead


def test_nonlinear_residual_learns_a_relation_absent_from_an_affine_readout():
    features = np.asarray([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
    labels = np.asarray([0., 1., 1., 0.])
    before = np.random.get_state()
    residual = fit_binding_residual(features, labels, np.ones(4), np.zeros(4),
                                    width=8, steps=400, seed=13)
    scores = np.asarray([residual.score(row) for row in features])
    assert np.array_equal(scores > 0., labels)
    assert all(np.array_equal(left, right) for left, right in zip(before, np.random.get_state(), strict=True))
    replay = NonlinearBindingResidual.from_dict(residual.to_dict())
    assert np.array_equal(scores, [replay.score(row) for row in features])
    for permutation in ([3, 1, 0, 2], [1, 0, 3, 2]):
        assert np.array_equal(scores[permutation], [replay.score(features[index]) for index in permutation])
    assert residual.scaled(0.).score(features[0]) == 0.
    assert not residual.kernel.flags.writeable


def test_triadic_nonlinear_score_and_all_lesions_round_trip():
    residual = NonlinearBindingResidual(np.zeros(8), np.ones(8), np.ones((8, 3)),
                                        np.zeros(3), np.ones(3), 2.)
    head = TriadicBindingHead(np.ones(8), 1., "joint_representation_v3", nonlinear=residual)
    op, mention, definition = np.asarray([1.]), np.asarray([2.]), np.asarray([3.])
    score = head.score(op, mention, definition)
    assert head.scaled(.5).score(op, mention, definition) == pytest.approx(score / 2.)
    assert head.score_lesion().score(op, mention, definition) == 0.
    for role in ("operation", "mention", "definition"):
        lesion = head.role_lesion(role)
        assert lesion.score(op, mention, definition) != score
    with pytest.raises(ValueError, match="width"):
        replace(head, nonlinear=replace(residual, center=np.zeros(7), scale=np.ones(7), kernel=np.ones((7, 3))))


def test_nonlinear_candidate_is_serialized_and_called_in_ordinary_decode(monkeypatch):
    from core.learning.semantic_program_compositional_refits import refit_compositional_triadic_bindings
    from core.learning.semantic_program_compositional_transducer import (
        compositional_semantic_program_transducer_from_dict, fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    with monkeypatch.context() as patch:
        patch.setattr(type(parent), "decode", lambda self, **kwargs: SimpleNamespace(ir=None, refusal="test"))
        candidate = refit_compositional_triadic_bindings(parent, examples,
            feature_schema="joint_representation_v3", nonlinear_width=8, nonlinear_steps=10)
    replay = compositional_semantic_program_transducer_from_dict(candidate.to_dict())
    assert replay.receipt_sha256 == candidate.receipt_sha256
    assert all(head.nonlinear is not None for head in replay.triadic_binding_heads)
    assert all(head.nonlinear is not None for head in replay.triadic_binding_lesion().triadic_binding_heads)
    calls = []
    original = NonlinearBindingResidual.score

    def observe(self, feature):
        calls.append(feature.copy())
        return original(self, feature)

    monkeypatch.setattr(NonlinearBindingResidual, "score", observe)
    item = next(value for value in examples if value.split == "test")
    replay.decode(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
                  public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
                  model_basis_sha256=replay.model_basis_sha256)
    assert calls


def test_fitted_triadic_evidence_enters_the_same_contextual_chart_solve():
    from core.learning.semantic_context_binding import (
        BindingRole, triadic_context_costs,
    )
    from tests.test_semantic_context_binding import chart, context

    roles = (BindingRole("left", "minuend", "Number"), BindingRole("right", "subtrahend", "Number"))
    weight = np.zeros(8)
    weight[5] = 1.
    head = TriadicBindingHead(weight, 0., "joint_representation_v3")
    operations = {role.identity: np.asarray([1.]) for role in roles}
    mentions = {"left": np.asarray([-1.]), "right": np.asarray([1.])}
    keys = {0: ("turn1", "result"), 1: ("turn2", "literal")}
    definitions = {keys[0]: np.asarray([2.]), keys[1]: np.asarray([-2.])}
    costs = triadic_context_costs(context(), roles, dict.fromkeys(mentions, head),
                                  operations, mentions, definitions)
    result = chart().solve_grounded(context(), (roles,), keys, costs=costs)
    assert result.status == "bound" and result.assignment[1] == ((1, 0),)
    assert triadic_context_costs(context(), roles, {}, operations, mentions, definitions) == {}
