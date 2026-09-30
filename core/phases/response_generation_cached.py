"""Serving a generation the turn already paid for, when the same request comes back.

Lifted whole out of `response_generation`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


async def _serve_the_cached_generation(
    *,
    generation_metadata: Any,
    latent_outcome: Any,
    latent_trace: Any,
    live_mind_controls_bound: Any,
    live_mind_generation_controls: Any,
    response_text: Any,
    self: Any,
    state: Any,
    token_budget: Any,
    visible_output_contract_payload: Any,
) -> tuple[Any, Any, Any]:
    """Serve a cached generation when this turn already has one.

    Moved out of ``ResponseGenerationPhase.execute`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 10 name(s) and hands back
    2.
    """
    from .response_generation import (
        _SEAM_FELL_THROUGH,
        logger,
    )

    async def _block() -> Any:
        nonlocal generation_metadata, response_text
        if latent_outcome.answer_available:
            response_text = latent_outcome.text
            latent_receipt = dict(
                latent_trace.get("latent_cortex_receipt") or {}
            )
            if not latent_outcome.succeeded:
                failure_reason = str(
                    latent_trace.get("latent_cortex_failure_reason")
                    or "latent_episode_failed"
                )
                state.response_modifiers.update(
                    {
                        "model_retry_suppressed": True,
                        "generation_failure_class": failure_reason[:120],
                    }
                )
            generation_metadata = {
                **latent_trace,
                "model_retry_suppressed": bool(
                    not latent_outcome.succeeded
                ),
                "surface_control_receipt": (
                    self._latent_cortex_surface_receipt(
                        latent_receipt,
                        controls_bound=live_mind_controls_bound,
                        generation_controls=live_mind_generation_controls,
                        token_budget=token_budget,
                        requested_output_contract=(
                            visible_output_contract_payload
                        ),
                    )
                ),
            }
        elif (
            latent_outcome.attempted
            and not latent_outcome.fallback_allowed
        ):
            failure_reason = str(
                latent_trace.get("latent_cortex_failure_reason")
                or "latent_owner_exhausted"
            )
            state.response_modifiers.update(
                {
                    "model_retry_suppressed": True,
                    "generation_failure_class": failure_reason[:120],
                    "response_path": (
                        "cognitive_engine_latent_owner_exhausted"
                    ),
                }
            )
            logger.error(
                "Recursive Latent Cortex exhausted the single resident "
                "owner (%s); refusing a colliding ordinary generation.",
                failure_reason,
            )
            return state
        return _SEAM_FELL_THROUGH

    _seam_early_response = await _block()
    return _seam_early_response, generation_metadata, response_text


