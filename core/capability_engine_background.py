"""Deferring a heavy background skill while live conversation resources are protected.

Lifted whole out of `capability_engine`, which imports them straight back: every
caller and every patch that names them there still finds them. What they
take from that module is imported at CALL time, for the same reason.
"""
from __future__ import annotations

from typing import Any


def _defer_a_heavy_background_skill(
    *,
    background_preflight_deferred: Any,
    ctx: Any,
    exec_source: Any,
    result: Any,
    self: Any,
    skill_name: Any,
) -> tuple[Any, Any, Any]:
    """Defer a heavy background skill while live conversation resources are protected.

    Moved out of ``_execute_wrapped`` by tools/extract_seam.py, which checks
    the body against the original token for token before writing. The
    block returns early, so it sits in a nested function and _SEAM_FELL_THROUGH
    means it finished instead. It reads 6 name(s) and hands back
    2.
    """
    from .capability_engine import (
        _HEAVY_BACKGROUND_SKILLS,
        _SEAM_FELL_THROUGH,
        _record_capability_degradation,
    )

    def _block() -> Any:
        nonlocal background_preflight_deferred, result
        if (
            not background_preflight_deferred
            and skill_name in _HEAVY_BACKGROUND_SKILLS
            and exec_source not in {"user", "api", "chat", "desktop", "voice", "web"}
        ):
            try:
                from core.runtime.background_policy import background_activity_reason

                reason = background_activity_reason(
                    ctx.get("orchestrator"),
                    min_idle_seconds=600.0,
                    max_memory_percent=70.0,
                    max_failure_pressure=0.20,
                    require_conversation_ready=False,
                )
                if reason:
                    background_preflight_deferred = True
                    result = {
                        "ok": False,
                        "status": "deferred",
                        "reason": reason,
                        "message": (
                            f"Background {skill_name} deferred while live conversation resources are protected ({reason})."
                        ),
                    }
            except (ImportError, AttributeError, RuntimeError) as policy_exc:
                _record_capability_degradation(
                    policy_exc,
                    action="deferred heavy background skill because preflight policy failed",
                    severity="degraded",
                )
                self.logger.warning(
                    "Heavy background preflight failed for %s: %s",
                    skill_name,
                    policy_exc,
                )
                return {
                    "ok": False,
                    "status": "deferred",
                    "reason": "background_policy_unavailable",
                    "message": (
                        f"Background {skill_name} deferred because the resource protection policy is unavailable."
                    ),
                }
        return _SEAM_FELL_THROUGH

    _seam_early_response = _block()
    return _seam_early_response, background_preflight_deferred, result


