"""What the response phase does for a proof or evaluation turn: the strict worker arguments, a retry against the answer options, and the repair of an incomplete evaluation.

Lifted whole out of `response_generation_unitary`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

import asyncio
from typing import Any


async def _retry_against_the_answer_options(
    *,
    _ensure_symbolic_consistency: Any,
    final_answer: Any,
    llm: Any,
    model_tier: Any,
    new_state: Any,
    objective: Any,
    option_present: Any,
    option_values: Any,
    request_timeout: Any,
    routing_origin: Any,
    self: Any,
    strict_envelope: Any,
    verify_system_prompt: Any,
) -> Any:
    """Retry once when the answer ignored the options the task supplied.

    Moved out of ``UnitaryResponsePhase.execute`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 12 name(s) from the turn and hands back
    1.
    """
    from .response_generation_unitary import (
        _RESPONSE_RECOVERABLE_ERRORS,
        _record_response_degradation,
        logger,
        mlx_strict_answer_contract_enabled,
    )

    if option_values and not option_present:
        option_system_prompt = (
            f"{verify_system_prompt} The original task provides answer options. "
            f"You must choose exactly one of these option values: {', '.join(option_values)}. "
            "Do not return the subject label or variable name."
        )
        option_messages = [
            {"role": "system", "content": option_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Original task:\n{objective}\n\n"
                    f"Invalid candidate answer:\n{final_answer}\n\n"
                    f"Choose exactly one option value from: {', '.join(option_values)}"
                ),
            },
        ]
        try:
            option_verified = await asyncio.wait_for(
                llm.think(
                    objective,
                    messages=option_messages,
                    system_prompt=option_system_prompt,
                    prefer_tier=model_tier,
                    deep_handoff=False,
                    allow_cloud_fallback=False,
                    origin=routing_origin,
                    purpose="strict_proof_answer_option_verify",
                    is_background=False,
                    foreground_request=True,
                    protected_foreground_lane=True,
                    strict_answer_contract=mlx_strict_answer_contract_enabled(
                        origin=routing_origin
                    ),
                    strict_value_contract=not mlx_strict_answer_contract_enabled(
                        origin=routing_origin
                    ),
                    skip_runtime_payload=True,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                    temperature=0.0,
                    max_tokens=64,
                    num_predict=64,
                    timeout=min(request_timeout, 90.0),
                    state=new_state,
                ),
                timeout=min(request_timeout, 90.0) + 5.0,
            )
        except _RESPONSE_RECOVERABLE_ERRORS as option_verify_exc:
            _record_response_degradation(
                option_verify_exc,
                "UnitaryResponse: strict proof option verification retry failed: %s",
                action="returned unverified option-shaped strict proof response after verifier failed",
                severity="warning",
            )
            option_verified = ""
        if isinstance(option_verified, dict):
            option_verified = (
                option_verified.get("content")
                or option_verified.get("response")
                or ""
            )
        option_envelope = self._coerce_strict_answer_envelope(option_verified)
        if option_envelope and self._strict_answer_value_allowed(
            objective,
            self._strict_answer_value_from_envelope(option_envelope),
            option_values=option_values,
        ):
            strict_envelope = option_envelope
            strict_envelope = await _ensure_symbolic_consistency(
                strict_envelope,
                stage="option_verifier",
            )
        elif option_envelope:
            logger.warning(
                "UnitaryResponse: rejected strict proof option verifier non-option: %r",
                self._strict_answer_value_from_envelope(option_envelope),
            )
        if not self._strict_answer_value_allowed(
            objective,
            self._strict_answer_value_from_envelope(strict_envelope),
            option_values=option_values,
        ):
            raise RuntimeError("strict_proof_option_contract_unmet")
    return strict_envelope


async def _repair_an_incomplete_proof_evaluation(
    *,
    contract: Any,
    llm: Any,
    model_tier: Any,
    new_state: Any,
    objective: Any,
    proof_evaluation_turn: Any,
    request_timeout: Any,
    response_text: Any,
    routing_origin: Any,
    self: Any,
) -> Any:
    """Repair a proof evaluation that stopped before it answered.

    Moved out of ``UnitaryResponsePhase.execute`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 10 name(s) from the turn and hands back
    1.
    """
    from .response_generation_unitary import (
        _RESPONSE_RECOVERABLE_ERRORS,
        _record_response_degradation,
        logger,
    )

    if proof_evaluation_turn and self._proof_evaluation_response_incomplete(
        objective,
        response_text,
    ):
        repair_system_prompt = self._build_proof_evaluation_system_prompt(
            new_state,
            contract,
        )
        repair_system_prompt = (
            "The previous proof/evaluation draft was incomplete or fragmentary. "
            "Regenerate a complete answer now. Use 3-6 complete sentences for explanation/planning tasks, "
            "answer only the current task, and do not mention this repair instruction.\n\n"
            f"{repair_system_prompt}"
        )
        repair_messages = [
            {"role": "system", "content": repair_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Current task:\n{objective}\n\n"
                    f"Incomplete draft:\n{response_text[:1200]}\n\n"
                    "Return a complete final response."
                ),
            },
        ]
        try:
            repaired = await asyncio.wait_for(
                llm.think(
                    objective,
                    messages=repair_messages,
                    system_prompt=repair_system_prompt,
                    prefer_tier=model_tier,
                    deep_handoff=False,
                    allow_cloud_fallback=False,
                    origin=routing_origin,
                    purpose="proof_evaluation_repair",
                    proof_evaluation_contract=True,
                    is_background=False,
                    foreground_request=True,
                    protected_foreground_lane=True,
                    skip_runtime_payload=True,
                    disable_prompt_cache=True,
                    clear_prompt_cache=True,
                    temperature=0.1,
                    max_tokens=768,
                    num_predict=768,
                    timeout=min(request_timeout, 120.0),
                    state=new_state,
                ),
                timeout=min(request_timeout, 120.0) + 5.0,
            )
            if isinstance(repaired, dict):
                repaired = repaired.get("content") or repaired.get("response") or ""
            repaired_text = str(repaired or "").strip()
            if repaired_text:
                response_text = repaired_text
                new_state.response_modifiers["proof_evaluation_repair"] = {
                    "reason": "incomplete_first_draft",
                    "complete": not self._proof_evaluation_response_incomplete(
                        objective,
                        repaired_text,
                    ),
                }
        except _RESPONSE_RECOVERABLE_ERRORS as proof_repair_exc:
            _record_response_degradation(
                proof_repair_exc,
                "UnitaryResponse: proof evaluation repair retry failed: %s",
                action="returned first proof evaluation draft after completeness repair failed",
                severity="warning",
            )
            logger.debug(
                "UnitaryResponse: proof evaluation repair retry failed: %s",
                proof_repair_exc,
            )
    return response_text


def _set_the_strict_proof_worker_arguments(
    *,
    llm_kwargs: Any,
    operator_evidence_turn: Any,
    proof_evaluation_turn: Any,
    routing_origin: Any,
    strict_proof_answer_request: Any,
    worker_strict_answer_contract: Any,
) -> None:
    """Set the worker arguments a strict proof answer needs.

    Moved out of ``UnitaryResponsePhase.execute`` by tools/extract_seam.py, which
    checks the body against the original token for token before
    writing. It reads 6 name(s) from the turn and hands back
    0.
    """
    if strict_proof_answer_request:
        llm_kwargs.update(
            {
                "purpose": "strict_proof_answer",
                "strict_answer_contract": worker_strict_answer_contract,
                "strict_value_contract": not worker_strict_answer_contract,
                "skip_runtime_payload": True,
                "disable_prompt_cache": True,
                "clear_prompt_cache": True,
                "temperature": 0.0,
                "max_tokens": 96,
                "num_predict": 96,
                "protected_foreground_lane": True,
            }
        )
    elif proof_evaluation_turn:
        llm_kwargs.update(
            {
                "purpose": "proof_evaluation",
                "proof_evaluation_contract": True,
                "skip_runtime_payload": True,
                "disable_prompt_cache": True,
                "clear_prompt_cache": True,
                "temperature": 0.1,
                "max_tokens": 640,
                "num_predict": 640,
                "protected_foreground_lane": True,
            }
        )
    elif operator_evidence_turn:
        llm_kwargs.update(
            {
                "purpose": "operator_evidence",
                "operator_evidence_contract": True,
                "skip_runtime_payload": True,
                "disable_prompt_cache": True,
                "clear_prompt_cache": True,
                "temperature": 0.1,
                "top_p": 0.8,
                "min_p": 0.03,
                "repetition_penalty": 1.18,
                "repetition_context_size": 96,
                "max_tokens": 220,
                "num_predict": 220,
                "protected_foreground_lane": True,
            }
        )
    elif routing_origin == "benchmark":
        llm_kwargs.update(
            {
                "purpose": "benchmark_evaluation",
                "benchmark_request": True,
                "proof_evaluation_contract": True,
                "proof_primary_lane_required": True,
                "skip_runtime_payload": True,
                "disable_prompt_cache": True,
                "clear_prompt_cache": True,
                "temperature": 0.1,
                "max_tokens": 2048,
                "num_predict": 2048,
                "protected_foreground_lane": True,
                "foreground_request": True,
                "is_background": False,
                "prefer_tier": "primary",
                "deep_handoff": False,
                "allow_deep_handoff": False,
                "allow_cloud_fallback": False,
            }
        )


