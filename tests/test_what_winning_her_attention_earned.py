"""What each source that won her attention has earned from the turns it won.

core/affect/what_winning_earned.py credits a turn's worth to the sources that
won the workspace during it, by their share of its wins, and the workspace
weighs their next bids by what they have earned.
"""

from __future__ import annotations

import time

import pytest

from core.affect import what_winning_earned as credit
from core.affect.what_winning_earned import CreditLedger


@pytest.fixture(autouse=True)
def fresh_ledger():
    credit.reset_for_test()
    yield
    credit.reset_for_test()


def test_a_source_that_keeps_winning_good_turns_earns_and_one_that_wins_bad_ones_loses():
    ledger = CreditLedger()
    wins = {"memory": 0, "drive": 0}
    for _ in range(6):
        wins["memory"] += 3
        ledger.note(dict(wins), 0.8)
        wins["drive"] += 3
        ledger.note(dict(wins), -0.8)
    assert ledger.earned("memory") > 0.5
    assert ledger.earned("drive") < -0.5


def test_a_turn_is_shared_by_how_much_of_it_each_source_held():
    ledger = CreditLedger()
    shares = ledger.note({"memory": 3, "drive": 1}, 1.0)
    assert shares == pytest.approx({"memory": 0.75, "drive": 0.25})


def test_only_the_wins_of_this_turn_count():
    ledger = CreditLedger()
    ledger.note({"memory": 10}, 1.0)
    shares = ledger.note({"memory": 10, "drive": 2}, -1.0)
    assert shares == {"drive": 1.0}


def test_a_turn_whose_worth_was_not_measured_credits_nobody_and_moves_the_count_on():
    ledger = CreditLedger()
    assert ledger.note({"memory": 5}, 0.0, measured=False) == {}
    assert ledger.earned("memory") == 0.0
    assert ledger.note({"memory": 6}, 1.0) == {"memory": 1.0}


def test_what_is_earned_grows_with_how_established_it_is():
    ledger = CreditLedger()
    ledger.note({"memory": 1}, 1.0)
    once = ledger.earned("memory")
    ledger.note({"memory": 2}, 1.0)
    twice = ledger.earned("memory")
    assert 0.0 < once < twice < 1.0


def test_a_count_or_amount_that_is_not_a_number_does_not_break_the_ledger():
    ledger = CreditLedger()
    assert ledger.note({"memory": "many", "drive": 2}, float("nan")) == {"drive": 1.0}
    assert ledger.earned("drive") == 0.0


def test_the_workspace_weighs_a_bid_by_what_its_source_has_earned():
    from core.consciousness.global_workspace import CognitiveCandidate, _bids

    ledger = credit.get_credit_ledger()
    for turn in range(1, 6):
        ledger.note({"earner": turn}, 1.0)
    now = time.time()
    earner = CognitiveCandidate(content="a", source="earner", priority=0.5)
    stranger = CognitiveCandidate(content="b", source="stranger", priority=0.5)
    bids = _bids([earner, stranger], {}, now)
    assert bids[id(earner)] > bids[id(stranger)]
