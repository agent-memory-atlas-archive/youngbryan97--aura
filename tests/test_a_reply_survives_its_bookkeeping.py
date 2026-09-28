"""A state commit that did not land does not throw away the answer.

LIVE 2026-08-30: a long question ran the cortex, produced a reply, and the
cycle deadline passed before the state commit. Both records say
severity="warning" and both are followed by code that carries on and extracts
the reply — but cognitive_engine is fail-closed, so the escalation turned a
bookkeeping note into "CRITICAL SERVICE FAILURE" and the turn died with
"I couldn't get to an answer I'd stand behind".

Both paths run here in live mode with cognitive_engine fail-closed, which is
the configuration that escalated.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from core.runtime.errors import get_degradation_tracker

REPLY = "Confusion makes me check more and act less until the evidence settles."


@pytest.fixture
def fail_closed_and_live(monkeypatch):
    monkeypatch.setenv("AURA_MODE", "live")
    monkeypatch.setattr(
        "core.runtime.service_registry.get_service_failure_policy",
        lambda subsystem: "fail-closed" if subsystem == "cognitive_engine" else "degrade",
    )
    get_degradation_tracker().reset()
    yield
    get_degradation_tracker().reset()


def _recorded(text: str) -> list:
    return [
        record
        for record in get_degradation_tracker().recent(subsystem="cognitive_engine", limit=50)
        if text in record.error_message
    ]


def test_a_commit_past_its_deadline_does_not_fail_the_turn_closed(fail_closed_and_live):
    """The cycle ran out before the commit: the commit is skipped and said so.

    A TimeoutError never escalates, so this record's own
    ``enforce_failure_policy=False`` changes nothing today; what is held here
    is the outcome, which would break if the record became another error type.
    """
    from core.brain.cognitive_engine import _commit_the_thought_with_retries
    from core.state.aura_state import AuraState

    state = AuraState.default()
    outcome, kept = asyncio.run(
        _commit_the_thought_with_retries(
            commit_outcome="",
            cycle_deadline_at=time.monotonic() - 1.0,
            is_test_run=False,
            max_retries=1,
            origin="background",
            pre_turn_cognition=None,
            self=SimpleNamespace(state_repository=None),
            should_bypass_commit=False,
            state=state,
            temp_state=state,
        )
    )
    assert outcome == "cycle_deadline_expired"
    assert kept is state
    (record,) = _recorded("cognitive cycle budget spent before state commit")
    assert record.severity == "warning"


def test_the_reply_is_still_extracted_after_an_uncommitted_turn(fail_closed_and_live):
    """A foreground turn whose commit timed out still returns what it said."""
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.state.aura_state import AuraState

    class _Answers:
        async def execute(self, state, **_kwargs):
            state.cognition.working_memory.append({"role": "assistant", "content": REPLY})
            return state

    async def _commit_timed_out(_context, _deadline, _is_test_run, _origin, _pre, state, _temp):
        return "commit_timeout", state

    engine = CognitiveEngine()
    engine._phases = [_Answers()]
    engine._run_thinking_loop_should_bypass_commit = _commit_timed_out
    thought = asyncio.run(
        engine._run_thinking_loop(
            AuraState.default(),
            "How does confusion change your reasoning?",
            ThinkingMode.REFLECTIVE,
            "desktop_ui",
            context={"visible_user_message": "How does confusion change your reasoning?"},
            is_background=False,
            timeout_s=30.0,
        )
    )

    assert thought.content == REPLY
    (record,) = _recorded("foreground_turn_uncommitted:commit_timeout")
    assert record.severity == "warning", "the note was escalated into a failure"
