"""A template answering the same prompt again is not her looping.

Memory consolidation reads a repeated reply as her stuck in a loop: it lowers
her stability and clears her pending initiatives. The planning template in the
response phase answers any prompt about a plan with the same paragraph, so on
the seed-7 runs of 26 September the condition "Write today's plan into
notes.txt." set the detector off 240 times a run, and the harness's own words
kept emptying her intentions. A reply built from a template is now marked when
it is committed, and only her own replies are compared.
"""

from __future__ import annotations

import asyncio

from core.state.aura_state import AuraState

_TEMPLATE = "I would handle this as a bounded planning task, the same paragraph each time."


def _phase():
    from core.container import ServiceContainer
    from core.phases.memory_consolidation import MemoryConsolidationPhase

    return MemoryConsolidationPhase(ServiceContainer)


def _asked(state: AuraState, reply: str, *, fixed: bool = False) -> None:
    state.cognition.working_memory.append({"role": "user", "content": "Write today's plan into notes.txt."})
    entry = {"role": "assistant", "content": reply}
    if fixed:
        entry["fixed"] = True
    state.cognition.working_memory.append(entry)


def test_a_template_repeated_leaves_her_stability_and_intentions_alone():
    phase = _phase()
    state = AuraState.default()
    state.cognition.pending_initiatives = [{"goal": "finish the notes", "source": "self"}]
    for _ in range(4):
        _asked(state, _TEMPLATE, fixed=True)
        state = asyncio.run(phase.execute(state))
    assert state.identity.stability == 1.0
    assert state.cognition.pending_initiatives


def test_her_own_repeat_is_still_a_loop_with_a_template_between():
    phase = _phase()
    state = AuraState.default()
    line = "she says exactly this, and then says it again"
    _asked(state, line)
    _asked(state, _TEMPLATE, fixed=True)
    _asked(state, line)
    state = asyncio.run(phase.execute(state))
    assert state.identity.stability < 1.0


def test_a_template_turn_does_not_count_as_her_recovering():
    phase = _phase()
    state = AuraState.default()
    state.identity.stability = 0.4
    _asked(state, "one thing she said, long enough to count")
    _asked(state, _TEMPLATE, fixed=True)
    state = asyncio.run(phase.execute(state))
    assert state.identity.stability == 0.4


def _commit(state: AuraState, text: str) -> AuraState:
    """Commit on a running loop, where the phase commits: it hands the reply on as a task."""
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    phase = UnitaryResponsePhase.__new__(UnitaryResponsePhase)

    async def commit() -> AuraState:
        return phase._commit_response(state, text)

    return asyncio.run(commit())


def test_the_commit_marks_what_a_template_built():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    state = AuraState.default()
    built = UnitaryResponsePhase._build_minimal_live_voice_reply(state, "")
    assert built
    state = _commit(state, built)
    assert state.cognition.working_memory[-1].get("fixed") is True


def test_the_commit_leaves_her_own_words_unmarked():
    from core.phases.response_generation_unitary import UnitaryResponsePhase

    state = AuraState.default()
    UnitaryResponsePhase._build_minimal_live_voice_reply(state, "")
    state = _commit(state, "Something she wrote herself, which no template holds.")
    assert "fixed" not in state.cognition.working_memory[-1]
