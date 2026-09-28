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
    assert source.count("_the_turn_is_waiting_on_this(kwargs)") == 3, (
        "every place the router decides foreground must read it, including the "
        "one the queueing reads"
    )


def test_the_gate_takes_a_bound_ledger_as_a_turn_in_flight():
    """The third account of an open turn, and the one a tool can always see."""
    from core.brain.inference_gate import _a_turn_ledger_is_bound

    assert _a_turn_ledger_is_bound() is False
    with bind_turn(TurnOutcome(origin="test")):
        assert _a_turn_ledger_is_bound() is True


def test_the_gate_reads_it_beside_its_own_accounts():
    import inspect

    from core.brain import inference_gate

    body = inspect.getsource(inference_gate)
    claim = body.split('context.get("serves_current_turn")', 1)[1][:1200]
    assert "_a_user_turn_is_in_flight()" in claim
    assert "_a_turn_ledger_is_bound()" in claim


def test_the_router_tells_the_endpoints_what_it_concluded():
    """The MLX clients keep their own mirror of the background policy.

    They cannot see a claim the router already authenticated, so the router
    admitted the browser's decision as foreground and the endpoint refused it
    with `foreground_quiet_window` a moment later — the page decision came back
    empty and the run stopped without clicking anything.
    """
    import inspect

    from core.brain import llm_health_router as router

    body = inspect.getsource(router)
    stamped = body.split("if explicit_foreground:", 1)[1][:800]
    assert 'kwargs["foreground_request"] = True' in stamped
