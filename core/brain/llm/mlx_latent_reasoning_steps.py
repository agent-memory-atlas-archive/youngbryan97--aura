"""Steps of `latent_reason_async`: the runtime controls on the job, the operation authority, and a deadline reached cleanly.

Lifted whole out of `mlx_latent_reasoning`, which imports them straight back: every
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


def _bind_the_runtime_controls_to_the_job(
    *,
    job: Any,
    runtime_controls: Any,
    wire_runtime_controls: Any,
) -> None:
    """Carry the live mind's runtime controls on the job, or allow full affective steering where there are none.

    Moved out of ``latent_reason_async`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 3 name(s) from the turn and hands back
    0.
    """
    if runtime_controls is not None:
        job["runtime_controls"] = wire_runtime_controls
        job["clean_user_surface_contract"] = True
        job["live_mind_controls_bound"] = True
        job.update(wire_runtime_controls)
    else:
        # Latent episodes without explicit surface-parity controls are
        # the experiment lane: they keep historical full governor
        # steering. Every OTHER worker job now defaults to the surface
        # clamp (fail-safe inversion after the July 2026 coherence
        # incident) — this opt-out is deliberately scoped to episodes.
        job["allow_full_affective_steering"] = True


def _say_the_owner_deadline_was_reached(
    *,
    progress: Any,
    receipt: Any,
) -> None:
    """Log a latent run that reached its owner's deadline cleanly, with where it stopped.

    Moved out of ``latent_reason_async`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 2 name(s) from the turn and hands back
    0.
    """
    from .mlx_latent_reasoning import (
        logger,
    )

    logger.warning(
        "Latent owner deadline reached cleanly: stage=%s "
        "input_tokens=%s elapsed=%s timings=%s",
        receipt.get("last_stage") or progress.get("stage") or "unknown",
        receipt.get("input_token_count")
        or progress.get("input_tokens")
        or "unknown",
        progress.get("elapsed_s") or "unknown",
        receipt.get("stage_timings_s") or {},
    )


def _check_the_operation_authority(
    *,
    base: Any,
    messages: Any,
    operation_authority: Any,
    prompt: Any,
    wire_action_policy_evidence: Any,
    wire_budget: Any,
    wire_cognitive_context: Any,
    wire_config: Any,
    wire_external_execution_offer: Any,
    wire_operation_authority: Any,
) -> tuple[Any, Any]:
    """Validate the runtime operation authority a latent run was given.

    Moved out of ``latent_reason_async`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 10 name(s) and hands back
    1.
    """
    # The marker latent_reason_async compares with, which it imports from
    # mlx_client. This read mlx_latent_reasoning's own marker, a different
    # object, so a check that passed came back as an early return and every
    # latent run answered with a bare object() (29 September to 1 October).
    from .mlx_client import (
        _SEAM_FELL_THROUGH,
    )

    def _block() -> Any:
        nonlocal wire_operation_authority
        if operation_authority is not None:
            try:
                from core.brain.llm.latent_cortex.epistemic_runtime import (
                    validate_runtime_operation_authority,
                )

                wire_operation_authority = validate_runtime_operation_authority(
                    operation_authority,
                    prompt=prompt,
                    messages=messages,
                    config=wire_config,
                    budget=wire_budget,
                    cognitive_context=wire_cognitive_context,
                    action_policy_evidence=wire_action_policy_evidence,
                    external_execution_offer=wire_external_execution_offer,
                )
            except (ImportError, TypeError, ValueError):
                return {**base, "reason": "invalid_runtime_operation_authority"}
        return _SEAM_FELL_THROUGH

    _seam_early_response = _block()
    return _seam_early_response, wire_operation_authority


