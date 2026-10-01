"""Tensor storage recovers supplied fillers; it does not determine their truth."""

import numpy as np
import pytest

from core.learning.semantic_binding_tensor import BindingTensor, tensor_from_grounded_binding
from core.learning.semantic_context_binding import (
    BindingContext,
    BindingRole,
    ContextReferent,
    bind_context_roles,
)


def test_exact_unbinding_does_not_require_distinct_or_independent_fillers():
    fillers = {"agent": [2., 3.], "patient": [2., 3.], "instrument": [4., 6.]}
    value = BindingTensor.pack(fillers)
    for role, expected in fillers.items():
        assert np.array_equal(value.unbind(role), expected)
    assert not value.tensor.flags.writeable


def test_nonorthogonal_roles_require_the_dual_not_the_raw_role_vectors():
    roles = np.asarray([[1., 0.], [1., 1.]])
    value = BindingTensor.pack({"a": [2., 3.], "b": [5., 7.]}, role_vectors=roles)
    assert not np.allclose(roles[0] @ value.tensor, [2., 3.])
    assert np.allclose(value.unbind("a"), [2., 3.])
    assert np.allclose(value.unbind("b"), [5., 7.])
    with pytest.raises(ValueError, match="independent"):
        BindingTensor.pack({"a": [1.], "b": [2.]}, role_vectors=np.ones((2, 2)))
    with pytest.raises(KeyError):
        value.unbind("unknown")


def test_only_accepted_source_observations_enter_tensor_storage():
    record = ContextReferent("chat", "a", "Object", "episode:0", embedding=(1., 2.))
    context = BindingContext((record,), {"Object": None})
    result = bind_context_roles(context, [BindingRole("role0", "theme", "Object", referents=(record.key,))])
    assert np.array_equal(tensor_from_grounded_binding(result, context).unbind("role0"), [1., 2.])
    unknown = bind_context_roles(context, [BindingRole("role0", "theme", "Object", referents=())])
    with pytest.raises(ValueError, match="unresolved"):
        tensor_from_grounded_binding(unknown, context)


def test_serialization_retains_distinct_sources_even_for_equal_fillers():
    tensor = BindingTensor.pack({"a": [1., 2.], "b": [1., 2.]},
        source_keys={"a": ("scene", "first"), "b": ("scene", "second")})
    replay = BindingTensor.from_dict(tensor.to_dict())
    assert replay.source_keys == tensor.source_keys
    assert replay.source_keys["a"] != replay.source_keys["b"]
    assert np.array_equal(replay.unbind("a"), replay.unbind("b"))
