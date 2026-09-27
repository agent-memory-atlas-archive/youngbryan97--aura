"""Arithmetic changes are explicit, reproducible, and cannot inherit authority."""

from copy import deepcopy

import mlx.core as mx
import mlx.nn as nn
import pytest

from core.learning.semantic_native_program import NativeProgramSequence
from tools.semantic_native_execution import (
    apply_execution,
    execution_contract,
    execution_from_plan,
    source_sequence_groups,
)


def test_historical_native_plans_do_not_change_arithmetic(monkeypatch):
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    model = nn.Linear(2, 2)
    model.set_dtype(mx.bfloat16)
    weight = model.weight
    assert execution_contract() is None
    assert execution_from_plan({}) is None
    assert apply_execution(model, {}) is None
    assert model.weight is weight


def test_trie_requires_the_measured_basis_and_launch_environment(monkeypatch):
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    with pytest.raises(ValueError, match="measured float32"):
        execution_contract(prefix_strategy="trie")
    with pytest.raises(ValueError, match="process launch"):
        execution_contract(precision="float32", prefix_strategy="trie")
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    contract = execution_contract(precision="float32", prefix_strategy="trie")
    assert execution_from_plan({"execution_contract": contract}, check_installed=True) == contract
    model = nn.Linear(2, 2)
    model.set_dtype(mx.bfloat16)
    assert apply_execution(model, {"execution_contract": contract})["after_dtype_counts"] == {
        "mlx.core.float32": 2}
    assert model.weight.dtype == mx.float32


@pytest.mark.parametrize("field,value", [("precision", "float16"), ("prefix_strategy", "auto"),
    ("serving_authority", True), ("qualification_evidence", True), ("schema", "unknown")])
def test_execution_refuses_relabelled_or_unsupported_contracts(monkeypatch, field, value):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    contract = execution_contract(precision="float32", prefix_strategy="trie")
    contract[field] = value
    with pytest.raises(ValueError, match="contract differs"):
        execution_from_plan({"execution_contract": contract})


def test_cpu_validation_and_model_replay_have_distinct_requirements(monkeypatch):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    contract = execution_contract(precision="float32")
    changed = deepcopy(contract)
    changed["installed_arithmetic"]["mlx_version"] = "different"
    monkeypatch.delenv("MLX_ENABLE_TF32")
    assert execution_from_plan({"execution_contract": contract}) == contract
    with pytest.raises(ValueError, match="process launch"):
        execution_from_plan({"execution_contract": contract}, check_installed=True)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    with pytest.raises(ValueError, match="installed arithmetic differs"):
        execution_from_plan({"execution_contract": changed}, check_installed=True)
    changed = deepcopy(contract)
    changed["precision_contract"]["target_logprob_tolerance"] *= 2
    with pytest.raises(ValueError, match="contract differs"):
        execution_from_plan({"execution_contract": changed})


def test_grouping_keeps_complete_source_choices_in_stable_order():
    row = NativeProgramSequence((1, 2, 3), 1)
    keys = [("b", 0, 0), ("a", 1, 1), ("a", 0, 0), ("a", 1, 0)]
    sequences = dict.fromkeys(keys, row)
    assert source_sequence_groups(sequences) == (
        (("a", 0, 0), ("a", 1, 0), ("a", 1, 1)), (("b", 0, 0),))
    assert source_sequence_groups(sequences) == source_sequence_groups(dict(reversed(list(sequences.items()))))
    with pytest.raises(ValueError, match="source-bound"):
        source_sequence_groups({"unbound": row})
