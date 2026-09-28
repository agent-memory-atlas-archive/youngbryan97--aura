"""A model call a user turn is waiting on is not background work.

LIVE 2026-09-28: "Queueing background inference until admission clears for
origin=sovereign_browser reason=foreground_chat_active". A tool the person asked
for makes model calls while their turn is open, and those calls carry the tool's
own origin, which is not a user-facing name. They were queued behind the very
turn that was waiting for them, the decision came back empty, and the run
stopped on the first page without clicking anything.

The flag alone would be an unauthenticated claim on the foreground lane, so it
counts only while a turn is genuinely bound.
"""
from __future__ import annotations

import pytest

from core.brain.llm_health_router import _the_turn_is_waiting_on_this
from core.runtime.turn_outcome import TurnOutcome, bind_turn

pytestmark = pytest.mark.unit


def test_it_counts_while_a_turn_is_bound():
    with bind_turn(TurnOutcome(origin="test")):
        assert _the_turn_is_waiting_on_this({"serves_current_turn": True}) is True


def test_outside_a_turn_it_means_nothing():
    assert _the_turn_is_waiting_on_this({"serves_current_turn": True}) is False


def test_a_request_that_makes_no_claim_is_unaffected():
    with bind_turn(TurnOutcome(origin="test")):
        assert _the_turn_is_waiting_on_this({}) is False
        assert _the_turn_is_waiting_on_this(None) is False
        assert _the_turn_is_waiting_on_this({"serves_current_turn": False}) is False


def test_the_router_reads_it_where_it_decides_foreground():
    import inspect

    from core.brain import llm_health_router as router

    source = inspect.getsource(router)
    assert source.count("_the_turn_is_waiting_on_this(kwargs)") == 2, (
        "both places the router decides foreground must read it"
    )
