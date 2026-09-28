"""The claims about her recurrent memory: the complete engine contract and the decode certificates, resident and at a path.

Lifted whole out of `model_validation`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _recurrent_memory_complete_engine_contract_holds() -> bool:
    from .model_validation import (
        NothingMeasured,
    )

    from core.learning.sealed_artifact_admission import mathematics_memory_admitted

    # A refused seal and a wrong answer are different facts, and this
    # predicate reported both as False. An unadmitted tissue never ran, so
    # NothingMeasured is the outcome — it keeps a provenance break from
    # reading as a capability regression. Measured 2026-08-15:
    # frontier_process_supervision.py drifted from its pinned hash (CP546).
    admitted, detail = mathematics_memory_admitted()
    if not admitted:
        raise NothingMeasured(
            "sealed mathematics memory tissue is not admitted, so the complete "
            f"engine never ran: {detail}"
        )

    from core.brain.llm.latent_cortex.frontier_tasks import generate_task
    from core.brain.llm.latent_cortex.neural_objective_producer import (
        solve_objective_program_neural,
    )
    from core.brain.llm.latent_cortex.objective_program_verifier import (
        verify_objective_program,
    )

    task = generate_task("mathematics", seed=1_037, difficulty=3)
    objective = task.public.prompt
    solved = solve_objective_program_neural(objective)
    if solved is None:
        return False
    candidate, receipt = solved
    verdict = verify_objective_program(candidate, objective=objective)
    execution = receipt.get("execution", {})
    student_rollin = execution.get("student_rollin", {})
    return bool(
        isinstance(execution, dict)
        and execution.get("engine") == "mathematics_memory_tissue.v1"
        and execution.get("teacher_available") is False
        and execution.get("independent_crosscheck_match") is True
        and isinstance(student_rollin, dict)
        and student_rollin.get("teacher_available") is False
        and student_rollin.get("verifier_available") is False
        and student_rollin.get("student_memory_rollin") is True
        and verdict is not None
        and verdict.get("outcome") == "verified"
    )


def _recurrent_memory_decode_certificate_holds() -> bool:
    from .model_validation import (
        _recurrent_memory_decode_certificate_holds_at,
    )

    return _recurrent_memory_decode_certificate_holds_at(
        "cp531_mathematics_memory_decode_verification.json"
    )


def _resident_recurrent_memory_decode_certificate_holds() -> bool:
    from .model_validation import (
        _recurrent_memory_decode_certificate_holds_at,
    )

    return _recurrent_memory_decode_certificate_holds_at(
        "cp534_resident_32b_mathematics_memory_decode_verification.json",
        expected_certificate_receipt=(
            "6dfe3e35e958412d0d4b737eb8e1d358038d1c4e560c4f83d479b6b8e62dd284"
        ),
        expected_artifact_receipt=(
            "8109dbe0e78651c55130b081639a1fbece53e49532087dcc8984a1ca03aa3b2b"
        ),
        expected_model_name="Qwen2.5-32B-Instruct-4bit",
        expected_config_sha=(
            "c027829d800805358d67ac87819a3754fd8240be973f7147840651310fd30ae3"
        ),
        expected_weights_index_sha=(
            "7b6da9b2b1f3ebd698ae15f9fcf6ba3099e742ec07e5d383f28b7cb77a4d16db"
        ),
        expected_claim_boundary=(
            "bounded teacher-free recurrent-state-to-free-decode transfer on "
            "the model cryptographically bound in model_identity; not "
            "open-domain, multi-domain, frontier-level, globally "
            "fusion-authorized, or WOW"
        ),
    )


def _recurrent_memory_decode_certificate_holds_at(
    certificate_name: str,
    *,
    expected_certificate_receipt: str | None = None,
    expected_artifact_receipt: str | None = None,
    expected_model_name: str | None = None,
    expected_config_sha: str | None = None,
    expected_weights_index_sha: str | None = None,
    expected_claim_boundary: str | None = None,
) -> bool:
    from .model_validation import (
        logger,
    )

    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    certificate_path = (
        root / "artifacts/closeout/latent_cortex" / certificate_name
    )
    try:
        certificate = json.loads(certificate_path.read_text(encoding="utf-8"))
        body = {
            key: value
            for key, value in certificate.items()
            if key != "receipt_sha256"
        }
        receipt = hashlib.sha256(
            json.dumps(
                body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        artifact_path = (root / certificate["artifact_path"]).resolve()
        if root not in artifact_path.parents:
            return False
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        artifact_body = {
            key: value for key, value in artifact.items() if key != "receipt_sha256"
        }
        artifact_receipt = hashlib.sha256(
            json.dumps(
                artifact_body,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
        ).hexdigest()
        verifier_path = root / "tools/verify_mathematics_memory_decode_canary.py"
        verifier_sha = hashlib.sha256(verifier_path.read_bytes()).hexdigest()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Recurrent memory decode certificate unreadable, claim unverified: %s", exc)
        return False
    controls = certificate.get("causal_control_exacts")
    base_contract_holds = bool(
        certificate.get("schema")
        == "aura.rlc.mathematics_memory_decode_canary_verification.v1"
        and certificate.get("receipt_sha256") == receipt
        and certificate.get("artifact_sha256") == artifact_sha
        and artifact.get("receipt_sha256") == artifact_receipt
        and certificate.get("verifier_source_sha256") == verifier_sha
        and certificate.get("verifier_source_clean") is True
        and certificate.get("independently_verified") is True
        and certificate.get("measurement_count") == 240
        and certificate.get("treatment_exact") == 30
        and certificate.get("ordinary_base_exact") == 0
        and certificate.get("matched_wire_base_exact") == 0
        and isinstance(controls, dict)
        and len(controls) == 6
        and set(controls.values()) == {0}
        and certificate.get("gain_count") == 30
        and certificate.get("regression_count") == 0
    )
    if not base_contract_holds:
        return False
    if (
        expected_certificate_receipt is not None
        and receipt != expected_certificate_receipt
    ):
        return False
    if (
        expected_artifact_receipt is not None
        and artifact_receipt != expected_artifact_receipt
    ):
        return False
    if expected_claim_boundary is not None and (
        certificate.get("claim_boundary") != expected_claim_boundary
        or artifact.get("claim_boundary") != expected_claim_boundary
    ):
        return False
    if expected_model_name is None:
        return True
    model_identity = artifact.get("model_identity")
    if not isinstance(model_identity, dict):
        return False
    model_path = model_identity.get("path")
    return bool(
        isinstance(model_path, str)
        and Path(model_path).name == expected_model_name
        and model_identity.get("config_sha256") == expected_config_sha
        and model_identity.get("weights_index_sha256")
        == expected_weights_index_sha
        and certificate.get("model_config_sha256") == expected_config_sha
    )


