"""An organ that supplied the winning bid earns from the turn, not only the winner.

A bid for her attention is a sum of parts from different organs: its own
priority, the urgency affect lends, where attention already is, the free-energy
engine's pull. Only the winning source was credited with a turn's worth, so an
organ whose part carried a win earned nothing however much the win depended on
it. Each win now records each organ's share of the winning bid, the ledger
credits it by that share, and what it has earned weighs its part of later bids
(core/consciousness/global_workspace_supply.py).
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.affect import what_winning_earned
from core.affect.what_winning_earned import CreditLedger
from core.consciousness.global_workspace import CognitiveCandidate, ContentType, GlobalWorkspace
from core.consciousness.global_workspace_supply import bid_parts, priority_of, supplied_shares


def _candidate(**extra) -> CognitiveCandidate:
    return CognitiveCandidate(
        content="a percept", source="perception", priority=0.4,
        content_type=ContentType.PERCEPTUAL, **extra,
    )


@pytest.fixture(autouse=True)
def fresh_ledger(monkeypatch):
    ledger = CreditLedger()
    monkeypatch.setattr(what_winning_earned, "_LEDGER", ledger)
    return ledger


def test_with_nothing_earned_a_bid_is_what_its_parts_add_to():
    candidate = _candidate(focus_bias=0.2, affect_weight=0.5)
    now = candidate.submitted_at
    parts, recency = bid_parts(candidate, now)
    expected = min(1.0, max(0.0, sum(parts.values()) * (0.7 + 0.3 * recency)))
    assert candidate.priority_at(now) == expected
    assert parts["bid"] == 0.4 and parts["attention"] == 0.2


def test_each_supplier_is_given_its_share_of_what_the_bid_was_made_of():
    shares = supplied_shares({"bid": 0.5, "attention": 0.3, "affect": 0.2, "relief": -0.1})
    assert shares == pytest.approx({"attention": 0.3, "affect": 0.2})


def test_a_win_records_what_attention_supplied():
    async def go() -> dict[str, float]:
        workspace = GlobalWorkspace()
        await workspace.submit(_candidate(focus_bias=0.3))
        await workspace.run_competition()
        return workspace.supplied_by_organ()

    supplied = asyncio.run(go())
    assert supplied.get("attention", 0.0) > 0.0


def test_a_good_turn_credits_the_organ_that_supplied_the_win(fresh_ledger):
    fresh_ledger.note({"perception": 0}, 0.0, supplied={"attention": 0.0})
    fresh_ledger.note({"perception": 1}, 0.8, supplied={"attention": 0.4, "affect": 0.1})
    assert fresh_ledger.supplier_earned("attention") > 0.0
    assert fresh_ledger.supplier_earned("affect") > 0.0
    assert "attention" not in fresh_ledger.read()["sources"]


def test_what_it_has_earned_weighs_its_part_of_later_bids(fresh_ledger):
    candidate = _candidate(focus_bias=0.2)
    now = time.time()
    before = priority_of(candidate, now)
    for turn in range(1, 6):
        fresh_ledger.note({"perception": turn}, 0.9, supplied={"attention": 0.3 * turn})
    assert fresh_ledger.supplier_earned("attention") > 0.0
    assert priority_of(candidate, now) > before


def test_a_turn_with_no_win_moves_the_supply_mark_on(fresh_ledger):
    fresh_ledger.note({"perception": 0}, 0.9, supplied={"attention": 0.5})
    fresh_ledger.note({"perception": 1}, 0.9, supplied={"attention": 0.6})
    # Only the 0.1 supplied during the second turn is that turn's.
    assert fresh_ledger.read()["suppliers"]["attention"]["seen"] == 1
