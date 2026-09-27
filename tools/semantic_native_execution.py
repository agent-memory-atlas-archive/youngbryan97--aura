"""Bind research training and replay to the same explicit decoder arithmetic."""

from __future__ import annotations

EXECUTION_PATHS = (
    "tools/semantic_native_execution.py",
    "tools/probe_semantic_native_prefix_branches.py",
    "core/learning/frozen_prefix_branches.py",
    "tools/probe_semantic_native_prefix_grouping.py",
)


def execution_contract(*, precision="native", prefix_strategy="full"):
    from tools.probe_semantic_native_prefix_branches import (
        installed_arithmetic_basis,
        precision_contract,
    )

    if precision not in {"native", "float32"} or prefix_strategy not in {"full", "trie"}:
        raise ValueError("unsupported native research execution")
    if prefix_strategy == "trie" and precision != "float32":
        raise ValueError("trie training requires its measured float32 arithmetic")
    if precision == "native":
        return None
    return {"schema": "aura.native_research_execution.v1", "precision": precision,
            "prefix_strategy": prefix_strategy, "precision_contract": precision_contract(precision),
            "installed_arithmetic": installed_arithmetic_basis(), "serving_authority": False,
            "qualification_evidence": False}


def execution_from_plan(plan, *, check_installed=False):
    contract = plan.get("execution_contract")
    if contract is None:
        return None
    if not isinstance(contract, dict):
        raise ValueError("native execution contract differs")
    # Contract validation is CPU-only and does not change the caller's environment.
    precision = contract.get("precision")
    strategy = contract.get("prefix_strategy")
    if (precision != "float32" or strategy not in {"full", "trie"}
            or contract.get("schema") != "aura.native_research_execution.v1"
            or contract.get("serving_authority") is not False
            or contract.get("qualification_evidence") is not False
            or not isinstance(contract.get("installed_arithmetic"), dict)
            or contract["installed_arithmetic"].get("MLX_ENABLE_TF32") != "0"):
        raise ValueError("native execution contract differs")
    basis = contract.get("precision_contract")
    expected = {"schema": "aura.native_branch_precision.v1", "mode": "float32",
                "floating_parameters": "float32", "packed_integer_weights": "unchanged",
                "reduced_precision_matmul": False, "target_logprob_tolerance": .015625,
                "tolerance_basis": "unchanged_native_bfloat16_probe_absolute_allowance",
                "native_full_sequence_reference_required": True,
                "native_ranking_change_grants_no_equivalence": True}
    if basis != expected or set(contract) != {"schema", "precision", "prefix_strategy",
            "precision_contract", "installed_arithmetic", "serving_authority", "qualification_evidence"}:
        raise ValueError("native execution contract differs")
    if check_installed and execution_contract(precision=precision, prefix_strategy=strategy) != contract:
        raise ValueError("native installed arithmetic differs from training")
    return contract


def apply_execution(model, plan):
    contract = execution_from_plan(plan, check_installed=True)
    if contract is None:
        return None
    from tools.probe_semantic_native_prefix_branches import convert_parameter_precision

    return convert_parameter_precision(model, mode=contract["precision"])


def source_sequence_groups(sequences):
    """Keep every alternative of each source together without reading labels."""
    groups = {}
    for key in sorted(sequences):
        if not isinstance(key, tuple) or not key or not isinstance(key[0], str):
            raise ValueError("native trie needs source-bound sequence identities")
        groups.setdefault(key[0], []).append(key)
    return tuple(tuple(keys) for keys in groups.values())
