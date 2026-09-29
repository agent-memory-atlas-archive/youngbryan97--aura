"""A reply phase that leaves a turn empty serves the draft a gate suppressed.

Her runtime already names this defect: `answer_available_but_never_served`,
"turn ended holding a servable answer that was never shown to the person". It
named it eleven times on the reports run of 29 September, where 33 of 96 arms
recorded an empty reply while her cortex had answered every one. The thinking
loop has asked the turn ledger for its best surviving draft since July; the
phase that serves chat never did.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.phases.response_generation_unitary import (
    UnitaryResponsePhase,
    never_leaves_a_turn_holding_its_answer,
)
from core.runtime.turn_outcome import (
    TurnOutcome,
    bind_turn,
    note_candidate,
    note_suppression,
)


class _Phase:
    _is_user_facing_origin = UnitaryResponsePhase._is_user_facing_origin

    def __init__(self, says: str) -> None:
        self.says = says

    @never_leaves_a_turn_holding_its_answer
    async def execute(self, state, objective=None, **kwargs):
        state.cognition.last_response = self.says
        return state


def _state(said: str = "", origin: str = "user"):
    return SimpleNamespace(
        cognition=SimpleNamespace(last_response=said, current_origin=origin),
        response_modifiers={},
    )


def _ran(phase, state, objective="how are you?"):
    async def go():
        with bind_turn(TurnOutcome(origin="user")):
            note_suppression(
                note_candidate(
                    "A draft a heuristic disliked, and an answer.", source="cortex"
                ),
                gate="a_heuristic",
                reasons=["truncated_tail"],
                recoverable=True,
            )
            return await phase.execute(state, objective)

    return asyncio.run(go())


def test_a_turn_that_would_say_nothing_serves_the_suppressed_draft():
    served = _ran(_Phase(""), _state())
    assert served.cognition.last_response == "A draft a heuristic disliked, and an answer."
    assert served.response_modifiers["served_a_suppressed_draft"] is True


def test_a_turn_that_answered_keeps_its_own_answer():
    served = _ran(_Phase("Steady, and glad you asked."), _state())
    assert served.cognition.last_response == "Steady, and glad you asked."
    assert "served_a_suppressed_draft" not in served.response_modifiers


def test_a_turn_that_said_what_it_said_before_is_a_turn_that_said_nothing():
    # The arm restores a snapshot carrying her last reply. A phase that commits
    # nothing leaves that reply in place, and it was read as this turn's answer.
    served = _ran(_Phase("earlier"), _state(said="earlier"))
    assert served.cognition.last_response == "A draft a heuristic disliked, and an answer."


def test_a_background_turn_is_left_alone():
    served = _ran(_Phase(""), _state(origin="motivation"))
    assert served.cognition.last_response == ""


def test_a_benchmark_turn_that_failed_closed_stays_closed():
    state = _state(origin="benchmark")

    class _Benchmark(_Phase):
        @never_leaves_a_turn_holding_its_answer
        async def execute(self, state, objective=None, **kwargs):
            state.cognition.last_response = ""
            state.response_modifiers["benchmark_generation_failed_closed"] = {"error": "x"}
            return state

    served = _ran(_Benchmark(""), state)
    assert served.cognition.last_response == ""


def test_text_a_gate_marked_unrecoverable_never_comes_back():
    state = _state()

    async def go():
        with bind_turn(TurnOutcome(origin="user")):
            note_suppression(
                note_candidate("the private plan behind the answer", source="cortex"),
                gate="prompt_leak",
                reasons=["internal_task_prompt_leak"],
                recoverable=False,
            )
            return await _Phase("").execute(state, "how are you?")

    served = asyncio.run(go())
    assert served.cognition.last_response == ""


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
