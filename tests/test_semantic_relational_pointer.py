"""Actual learned pointer evidence reaches the existing contextual graph solve."""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from core.learning.semantic_context_binding import (
    BindingContext,
    BindingRelation,
    BindingRole,
    ContextReferent,
    bind_context_roles,
)
from core.learning.semantic_relational_pointer import (
    RelationalBindingPointer,
    pointer_choice_loss,
    pointer_context_costs,
)


def evidence():
    operations = mx.array([[[1., 0., 0., 1.], [2., 0., 0., 1.]],
                           [[0., 1., 1., 0.], [0., 2., 1., 0.]]])
    mentions = operations * .5
    candidates = mx.array([[[0., 1., 0., 0.], [0., 2., 0., 0.]],
                           [[1., 0., 0., 0.], [2., 0., 0., 0.]],
                           [[0., 0., 1., 1.], [0., 0., 2., 1.]]])
    return operations, mentions, candidates


def trained_pointer():
    mx.random.seed(13)
    pointer = RelationalBindingPointer(4, depths=2, relation_width=8, rounds=2, role_queries=False)
    inputs = evidence()
    optimizer = optim.Adam(learning_rate=.02)
    initial = pointer_choice_loss(pointer(*inputs), ((1,), (0,))).item()
    for _ in range(25):
        loss, grads = nn.value_and_grad(pointer, lambda model:
            pointer_choice_loss(model(*inputs), ((1,), (0,))))(pointer)
        optimizer.update(pointer, grads)
        mx.eval(pointer.parameters(), loss)
    assert pointer_choice_loss(pointer(*inputs), ((1,), (0,))).item() < initial / 4
    return pointer


def test_real_gradient_learning_and_identity_conditioned_solver_integration():
    pointer = trained_pointer()
    operations, mentions, candidates = evidence()
    records = tuple(ContextReferent("turn", str(index), "Number", "observed:" + str(index))
                    for index in range(3))
    context = BindingContext(records, {"Number": None})
    roles = (BindingRole("lhs", "minuend", "Number"),
             BindingRole("rhs", "subtrahend", "Number"))
    costs = pointer_context_costs(pointer, context, roles,
        {role.identity: operations[index] for index, role in enumerate(roles)},
        {role.identity: mentions[index] for index, role in enumerate(roles)},
        {record.key: candidates[index] for index, record in enumerate(records)})
    result = bind_context_roles(context, roles, costs=costs,
                               relations=(BindingRelation("lhs", "rhs", "different"),))
    assert result.executable
    assert result.bindings == (("lhs", ("turn", "1")), ("rhs", ("turn", "0")))


def test_candidate_and_role_permutations_commute_with_graph_refinement():
    pointer = trained_pointer()
    operations, mentions, candidates = evidence()
    adjacency = mx.array([[0., 0., 1., 2., 0.], [0., 0., 0., 1., 1.],
                          [1., 0., 0., 0., 1.], [2., 1., 0., 0., 0.],
                          [0., 1., 1., 0., 0.]])
    expected = pointer(operations, mentions, candidates, adjacency=adjacency)
    role_order, candidate_order = mx.array([1, 0]), mx.array([2, 0, 1])
    node_order = mx.array([1, 0, 4, 2, 3])
    actual = pointer(operations[role_order], mentions[role_order], candidates[candidate_order],
                     adjacency=adjacency[node_order][:, node_order])
    assert mx.allclose(actual, expected[role_order][:, candidate_order], atol=1e-5).item()
    assert not mx.allclose(expected, pointer(operations, mentions, candidates)).item()


def test_zero_graph_has_no_invented_messages_and_initial_residual_is_zero():
    pointer = RelationalBindingPointer(4, depths=2, role_queries=False)
    operations, mentions, candidates = evidence()
    assert mx.array_equal(pointer(operations, mentions, candidates), mx.zeros((2, 3))).item()
    pointer = trained_pointer()
    assert mx.array_equal(pointer(operations, mentions, candidates, adjacency=mx.zeros((5, 5))),
                          pointer(operations, mentions, candidates)).item()


def test_missing_evidence_and_invalid_graph_are_not_silently_filled():
    pointer = RelationalBindingPointer(4, depths=2, role_queries=False)
    for bad in (mx.full((5, 5), -1.), mx.full((5, 5), mx.nan), mx.zeros((4, 4))):
        with pytest.raises(ValueError, match="graph evidence|adjacency"):
            pointer(*evidence(), adjacency=bad)
    with pytest.raises(ValueError, match="incomplete"):
        pointer_context_costs(pointer,
            BindingContext((ContextReferent("x", "1", "Number", "t:1"),), {"Number": None}),
            (BindingRole("r", "value", "Number"),), {}, {}, {})


def test_masked_positives_cannot_train_as_legal_bindings():
    pointer = RelationalBindingPointer(4, depths=2, role_queries=False)
    mask = mx.array([[True, False, True], [True, True, True]])
    scores = pointer(*evidence(), allowed=mask)
    assert mx.isneginf(scores[0, 1]).item()
    with pytest.raises(ValueError, match="admitted finite"):
        pointer_choice_loss(scores, ((1,), (0,)))
    assert mx.isfinite(pointer_choice_loss(scores, ((0, 2), (0,)))).item()
    assert pointer_choice_loss(mx.array([[2.]]), ((0,),)).item() == 0.


def test_checkpoint_roundtrip_preserves_all_depth_and_graph_parameters(tmp_path):
    pointer = trained_pointer()
    path = tmp_path / "pointer.safetensors"
    pointer.save_weights(str(path))
    restored = RelationalBindingPointer(4, depths=2, relation_width=8, rounds=2, role_queries=False)
    restored.load_weights(str(path), strict=True)
    assert mx.array_equal(pointer(*evidence()), restored(*evidence())).item()
    assert restored.to_contract() == pointer.to_contract()


def test_logical_operand_queries_distinguish_identical_occurrence_evidence_and_remain_equivariant():
    from core.learning.semantic_context_binding import BindingRole
    from core.learning.semantic_relational_pointer import semantic_role_features

    mx.random.seed(52)
    pointer = RelationalBindingPointer(4, relation_width=16, rounds=0)
    operation = mx.array([[[1., 0., 1., 0.]], [[1., 0., 1., 0.]]])
    mention = mx.array([[[0., 1., 0., 1.]], [[0., 1., 0., 1.]]])
    candidates = mx.array([[[1., 0., 0., 0.]], [[0., 1., 0., 0.]]])
    roles = tuple(BindingRole(f"instance:{slot}", f"sub:operand:{slot}", "integer") for slot in range(2))
    queries = semantic_role_features(roles)
    optimizer = optim.Adam(learning_rate=.03)
    for _ in range(32):
        loss, grads = nn.value_and_grad(pointer, lambda owner:
            pointer_choice_loss(owner(operation, mention, candidates, role_features=queries), ((1,), (0,))))(pointer)
        optimizer.update(pointer, grads)
        mx.eval(pointer.parameters(), loss)
    scores = pointer(operation, mention, candidates, role_features=queries)
    assert tuple(mx.argmax(scores, axis=-1).tolist()) == (1, 0)
    lesioned = pointer(operation, mention, candidates, role_features=mx.zeros_like(queries))
    assert mx.array_equal(lesioned[0], lesioned[1]).item()
    reversed_roles = pointer(operation, mention, candidates, role_features=queries[mx.array([1, 0])])
    assert tuple(mx.argmax(reversed_roles, axis=-1).tolist()) == (0, 1)
    permutation = mx.array([1, 0])
    permuted = pointer(operation[permutation], mention[permutation], candidates[permutation],
                       role_features=queries[permutation])
    assert mx.allclose(permuted, scores[permutation][:, permutation]).item()
    with pytest.raises(ValueError, match="declared operation operand"):
        semantic_role_features((BindingRole("bad", "sub:operand:7", "integer"),))


def test_operation_specific_roles_and_type_queries_are_not_collapsed():
    from core.learning.semantic_relational_pointer import semantic_role_features

    roles = (BindingRole("a", "sub:operand:0", "integer"),
             BindingRole("b", "idiv:operand:0", "integer"))
    queries = semantic_role_features(roles)
    assert not mx.array_equal(queries[0], queries[1]).item()
    with pytest.raises(ValueError, match="type differs"):
        semantic_role_features((BindingRole("bad", "at:operand:0", "integer"),))
    with pytest.raises(ValueError, match="declared role"):
        semantic_role_features((BindingRole("bad", "unobserved_role", "Number"),))
    pointer = RelationalBindingPointer(4, depths=2)
    with pytest.raises(ValueError, match="requires explicit"):
        pointer(*evidence())


def test_role_orthogonality_does_not_guarantee_truth_and_margins_keep_equivalent_positives():
    from core.learning.semantic_relational_pointer import (
        pointer_role_margin_loss,
        semantic_role_features,
    )

    pointer = RelationalBindingPointer(4, depths=2)
    queries = semantic_role_features((BindingRole("a", "minuend", "Number"),
                                      BindingRole("b", "subtrahend", "Number")))
    scores = pointer(*evidence(), role_features=queries)
    assert mx.array_equal(scores, mx.zeros((2, 3))).item()
    assert pointer_role_margin_loss(mx.array([[2., 2., -1.]]), ((0, 1),), margin=.5).item() == 0.
    assert pointer_role_margin_loss(mx.array([[0., 1., -mx.inf]]), ((0,),), margin=.5).item() == 1.5
    with pytest.raises(ValueError):
        pointer_role_margin_loss(scores, ((0,), (0,)), margin=-1.)
