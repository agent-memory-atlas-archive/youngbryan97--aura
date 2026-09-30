"""Verify retained optimizer evidence against the coefficients a model carries."""

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from core.learning.semantic_fit_checkpoint import fit_identity, read_fit_archive


def binary_model_heads(model: Any) -> tuple[tuple[str, Any, float], ...]:
    heads = []
    for name in ("operation_pointer", "argument_pointer", "definition_pointer"):
        pointer = getattr(model, name)
        for boundary in ("start", "end"):
            heads.append((f"{name}.{boundary}", getattr(pointer, boundary + "_weight"),
                          getattr(pointer, boundary + "_bias")))
    for name in ("argument_role_heads", "argument_proposal_heads"):
        heads.extend((f"{name}.{index}", head.weight, head.bias)
                     for index, head in enumerate(getattr(model, name)))
    relation = model.definition_relation_head
    heads.append(("definition_relation_head", relation.weight, relation.bias))
    return tuple(heads)


def verify_binary_fit_checkpoints(model: Any, directory: Path) -> dict[str, Any]:
    """Read every archive independently; a status string is not sufficient."""
    receipt = model.training_receipt.get("binary_head_fit_checkpoints")
    if not isinstance(receipt, dict):
        raise ValueError("source model has no measured binary checkpoint inventory")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    heads = binary_model_heads(model)
    records, archives = receipt.get("records"), receipt.get("final_archives")
    if (receipt.get("schema") != "aura.semantic_binary_fit_checkpoints.v1"
            or receipt.get("receipt_sha256") != fit_identity(body)
            or receipt.get("cached_convergence_is_authority") is not False
            or receipt.get("serving_authority") is not False
            or not isinstance(records, list) or len(records) != len(heads)
            or not isinstance(archives, dict)
            or set(archives) != {row.get("objective_identity") for row in records}):
        raise ValueError("binary checkpoint receipt or head inventory differs")
    checked = []
    for (name, weight, bias), row in zip(heads, records, strict=True):
        identity = row.get("objective_identity")
        if (not isinstance(identity, str) or len(identity) != 64
                or any(char not in "0123456789abcdef" for char in identity)
                or row.get("converged") is not True
                or any(type(row.get(key)) is not int or row[key] < 0
                       for key in ("iterations", "resumed_iterations"))
                or type(row.get("max_iter")) is not int or row["max_iter"] < row["iterations"]
                or not math.isfinite(row.get("objective", float("nan")))
                or not math.isfinite(row.get("tolerance", float("nan")))
                or row["tolerance"] <= 0):
            raise ValueError("binary convergence record is incomplete")
        path = Path(directory) / (identity + ".npz")
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != archives[identity]:
            raise ValueError("binary final archive identity differs")
        metadata, arrays = read_fit_archive(path, ("weight", "center"))
        actual = arrays["weight"]
        expected = np.asarray(weight, dtype=np.float32).reshape(-1)
        if (metadata.get("schema") != "aura.objective_fit_checkpoint.v1"
                or metadata.get("identity") != identity
                or metadata.get("status") != "converged"
                or metadata.get("iterations") != row["iterations"] + row["resumed_iterations"]
                or any(value.dtype != np.float64 or value.shape != (expected.size + 1,)
                       or not np.all(np.isfinite(value)) for value in arrays.values())
                or np.any(arrays["center"] != 0)
                or fit_identity(actual) != row.get("final_parameters_sha256")
                or not np.array_equal(actual[:-1].astype(np.float32), expected)
                or float(actual[-1]) != float(bias)):
            raise ValueError(f"binary archive does not carry the model head: {name}")
        checked.append({"head": name, "objective_identity": identity,
                        "archive_sha256": archives[identity],
                        "fresh_iterations": row["iterations"],
                        "resumed_iterations": row["resumed_iterations"],
                        "coefficient_matches": True})
    body = {"schema": "aura.semantic_binary_fit_verification.v1",
            "candidate_receipt_sha256": model.receipt_sha256,
            "checkpoint_receipt_sha256": receipt["receipt_sha256"], "heads": checked,
            "all_archives_verified": True, "all_coefficients_match": True,
            "objective_recomputed": False, "semantic_accuracy_measured": False,
            "qualification_evidence": False, "serving_authority": False}
    return {**body, "receipt_sha256": fit_identity(body)}
