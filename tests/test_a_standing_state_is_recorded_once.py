"""Withholding relational memory is a state of a request path, not an event.

The reports ground of 29 September recorded
`relational memory withheld: no bound principal` 332 times in one run, with a
fault beside each, because nothing binds a principal outside the chat routes:
`interface/routes/chat.py` opens `relational_principal_scope`, and a
subject-core driver calls `turn_once` directly. Withholding was correct every
time. Recording it 332 times is 331 records of nothing new, and a real fault is
harder to find among them.
"""

from __future__ import annotations

import pytest

import core.brain.llm.context_assembler_blocks as blocks
from core.brain.llm.context_assembler_blocks import (
    _BuildsThePromptBlocks,
    forget_withheld_origins_for_test,
)

pytestmark = pytest.mark.unit


class _State:
    def __init__(self):
        self.response_modifiers: dict = {}


@pytest.fixture(autouse=True)
def _forget():
    forget_withheld_origins_for_test()
    yield
    forget_withheld_origins_for_test()


@pytest.fixture
def recorded(monkeypatch):
    seen: list[tuple[str, str]] = []
    import core.brain.llm.context_assembler as assembler

    monkeypatch.setattr(
        assembler,
        "record_degradation",
        lambda subsystem, exc, **_kw: seen.append((subsystem, str(exc))),
    )
    return seen


def _withhold(origin="desktop"):
    return _BuildsThePromptBlocks._build_system_prompt_agent_id(
        bound_agent="", hinted_agent="bryan", internal_unbound_scope=False,
        request_origin=origin, state=_State(),
    )


def test_a_turn_with_a_hint_and_no_principal_says_so(recorded):
    _withhold()
    assert len(recorded) == 1
    assert recorded[0][0] == "context_assembler.relational_scope"
    assert "no bound principal" in recorded[0][1]


def test_the_next_three_hundred_turns_say_nothing_new(recorded):
    for _ in range(332):
        _withhold()
    assert len(recorded) == 1


def test_each_request_path_is_told_for_itself(recorded):
    _withhold("desktop")
    _withhold("voice")
    _withhold("desktop")
    assert len(recorded) == 2
    assert [message.split(" for ")[1].split(" requests")[0] for _, message in recorded] == [
        "desktop",
        "voice",
    ]


def test_a_path_that_binds_a_principal_is_told_again_if_it_stops(recorded):
    _withhold("desktop")
    _BuildsThePromptBlocks._build_system_prompt_agent_id(
        bound_agent="bryan", hinted_agent="bryan", internal_unbound_scope=False,
        request_origin="desktop", state=_State(),
    )
    _withhold("desktop")
    assert len(recorded) == 2


def test_an_internal_unbound_scope_still_writes_its_receipt(recorded):
    state = _State()
    _BuildsThePromptBlocks._build_system_prompt_agent_id(
        bound_agent="", hinted_agent="", internal_unbound_scope=True,
        request_origin="motivation", state=state,
    )
    assert state.response_modifiers["relational_scope_receipt"]["principal_bound"] is False
    assert not recorded


def test_the_memory_is_still_withheld(recorded):
    agent_id, relational_block = _withhold()
    assert agent_id == ""
    assert relational_block == ""


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
