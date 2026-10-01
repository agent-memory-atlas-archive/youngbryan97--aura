"""The contract predicates for the cortex's governed tools and the confined Python runner.

Lifted whole out of `model_validation`, which imports them straight back: every
caller and every patch that names them there still finds them. Each imports
what it reads at call time, as it did there.
"""
from __future__ import annotations

from core.organism.nothing_measured import NothingMeasured


def _rlc_capability_evidence_contract_holds() -> bool:
    import hashlib

    from core.brain.capability_evidence_context import (
        build_current_turn_capability_evidence,
    )

    objective = "Use Python to calculate the exact checksum total."
    objective_sha256 = hashlib.sha256(objective.encode("utf-8")).hexdigest()
    admitted = build_current_turn_capability_evidence(
        {
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": objective_sha256,
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "checksum_total=4182",
                "exit_code": 0,
            },
        },
        objective,
    )
    stale = build_current_turn_capability_evidence(
        {
            "last_skill_run": "run_code",
            "last_skill_ok": True,
            "last_skill_objective_hash": "0" * 64,
            "last_skill_result_payload": {
                "ok": True,
                "stdout": "stale=1",
                "exit_code": 0,
            },
        },
        objective,
    )
    return bool(
        admitted.receipt.get("admitted") is True
        and len(admitted.items) == 1
        and admitted.items[0].get("instruction_authority") is False
        and admitted.items[0].get("evidence_kind") == "governed_tool_observation"
        and not stale.items
        and stale.receipt.get("reason") == "stale_skill_result"
    )


def _rlc_web_acquisition_contract_holds() -> bool:
    from core.brain.cortex_web_acquisition import should_acquire_live_web
    from core.brain.llm.latent_cortex.context_focus import source_matches_action
    from core.executive.standing_authority import AUTONOMOUS_AUTHORITY_ORIGINS

    live = should_acquire_live_web(
        "What is the latest compiler release?",
        "compiler release",
        local_context_is_new=True,
    )
    uncovered = should_acquire_live_web(
        "Explain the new theorem.",
        "new theorem",
        local_context_is_new=False,
    )
    return bool(
        live == (True, "live_or_source_sensitive_objective")
        and uncovered == (True, "local_reference_uncovered")
        and "latent_cortex" in AUTONOMOUS_AUTHORITY_ORIGINS
        and source_matches_action("capability.web_search", "retrieve_evidence")
    )


def _rlc_amplifier_composition_contract_holds() -> bool:
    from core.brain.reasoning_amplifier_v2 import _admit_seed_candidates

    return _admit_seed_candidates(
        ["candidate", "candidate", ""],
        limit=2,
    ) == ["candidate"]


def _symbolic_cognition_boundary_available() -> bool:
    from core.sandbox.untrusted_python import available_boundary

    return available_boundary() in {"seatbelt", "bubblewrap"}


def _sandbox_async_execution_probe() -> bool:
    from core.sandbox.untrusted_python import (
        available_boundary,
        call_untrusted_function,
        run_untrusted_script,
    )

    if not available_boundary():
        raise NothingMeasured("no kernel sandbox is available for the async execution probe")
    completed = call_untrusted_function(
        "async def answer():\n    return 42\n", "answer", [()],
        timeout_s=2.0, source="validation.async_execution",
    )
    dropped = run_untrusted_script(
        "async def answer():\n    return 42\nanswer()\n",
        timeout_s=2.0, source="validation.async_execution",
    )
    if completed.status not in {"ok", "error"} or dropped.status not in {"ok", "error"}:
        raise NothingMeasured("the async execution probe did not finish under its kernel boundary")
    return bool(
        completed.ok and completed.results == [42]
        and dropped.status == "error" and "never awaited" in dropped.error
    )


def _rlc_compute_continuation_contract_holds() -> bool:
    from core.brain.llm.latent_cortex.cognitive_acquisition import (
        acquisition_has_new_context,
        build_acquisition_request,
    )

    transition = {
        "action": "formalize",
        "outcome": "succeeded",
        "checked": True,
    }
    request = build_acquisition_request(
        objective="Compute 12 * 13 exactly.",
        first_text="The answer is 157.",
        first_receipt={
            "cognitive_action_trace": [
                {"decision": {"action": "formalize"}, "transition": transition}
            ]
        },
        cognitive_context=None,
    )
    return bool(
        request
        and request.get("action") == "formalize"
        and request.get("max_acquisitions") == 1
        and request.get("max_continuation_rounds") == 1
        and acquisition_has_new_context(
            request,
            [
                {
                    "source": "capability.symbolic_formalize",
                    "text": "exact(12*13) = 156",
                }
            ],
        )
    )
