"""A new arithmetic basis cannot relabel the rejected native cache experiment."""

import ast
import inspect

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from tools.probe_semantic_native_prefix_branches import (
    float32_parameters,
    installed_arithmetic_basis,
    precision_contract,
    target_logprobs,
)


def test_native_precision_keeps_the_original_contract(monkeypatch):
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    assert precision_contract("native") is None
    for mode in ("float32", "unknown", None):
        with pytest.raises(ValueError, match="process launch"):
            precision_contract(mode)
    monkeypatch.setenv("MLX_ENABLE_TF32", "1")
    with pytest.raises(ValueError, match="process launch"):
        precision_contract("float32")


def test_float32_requires_full_native_reference_and_unchanged_error_allowance(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    contract = precision_contract("float32")
    assert contract["target_logprob_tolerance"] == .015625
    assert contract["native_full_sequence_reference_required"] is True
    assert contract["native_ranking_change_grants_no_equivalence"] is True
    assert contract["packed_integer_weights"] == "unchanged"
    assert contract["reduced_precision_matmul"] is False


def test_parameter_conversion_retains_packed_quantized_weights_and_values():
    mx.random.seed(20260926)
    model = nn.Sequential(nn.Linear(32, 32), nn.RMSNorm(32))
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=32, bits=4)
    before = dict(tree_flatten(model.parameters()))
    packed = {name: value for name, value in before.items() if not mx.issubdtype(value.dtype, mx.floating)}
    values = {name: value.tolist() for name, value in packed.items()}
    receipt = float32_parameters(model)
    after = dict(tree_flatten(model.parameters()))
    assert packed
    assert receipt["unchanged_nonfloating_parameters"] == len(packed)
    assert receipt["after_parameter_bytes"] > receipt["before_parameter_bytes"]
    assert all(after[name] is value and after[name].tolist() == values[name] for name, value in packed.items())
    assert all(value.dtype == mx.float32 for value in after.values() if mx.issubdtype(value.dtype, mx.floating))
    assert model(mx.ones((1, 3, 32), dtype=mx.float32)).dtype == mx.float32


def test_precision_conversion_refuses_inventory_changes():
    class BrokenCast(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.ones((2, 2))

        def set_dtype(self, dtype):
            self.unexpected = mx.zeros((2, 2))

    with pytest.raises(ValueError, match="inventory"):
        float32_parameters(BrokenCast())
    with pytest.raises(ValueError, match="declared model parameters"):
        float32_parameters(nn.Module())


def test_precision_conversion_refuses_replacement_of_packed_weights():
    class BrokenCast(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = mx.ones((2, 2), dtype=mx.uint32)

        def set_dtype(self, dtype):
            self.weight = mx.array(self.weight)

    with pytest.raises(ValueError, match="packed integer"):
        float32_parameters(BrokenCast())


def test_logprob_comparison_uses_float32_normalization_at_every_target():
    logits = mx.array([[[1., 2., 3.], [4., 2., -1.]]], dtype=mx.bfloat16)
    targets = mx.array([2, 0])
    actual = target_logprobs(logits, targets)
    expected = logits[0].astype(mx.float32)
    expected = mx.take_along_axis(expected, targets[:, None], axis=1)[:, 0] - mx.logsumexp(expected, axis=-1)
    assert actual.dtype == mx.float32
    assert mx.array_equal(actual, expected).item()


def test_precision_basis_pins_the_installed_arithmetic(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    basis = installed_arithmetic_basis()
    assert basis["MLX_ENABLE_TF32"] == "0"
    assert basis["mlx_version"] and basis["mlx_lm_version"]
    assert len(basis["implementation"]) == 4
    assert all(len(value) == 64 for value in basis["implementation"].values())


def test_probe_requires_exclusive_ownership_without_evicting_other_work():
    from tools.probe_semantic_native_prefix_branches import main

    tree = ast.parse(inspect.getsource(main))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "standalone_model_lane"]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.literal_eval(keywords["require_exclusive"]) is True
    assert ast.literal_eval(keywords["allow_owner_eviction"]) is False
