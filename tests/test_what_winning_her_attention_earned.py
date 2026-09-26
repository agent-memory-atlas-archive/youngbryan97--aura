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
    """Against turns that paid nothing, a source whose turns paid well stands out more each time."""
    ledger = CreditLedger()
    wins = {"memory": 0, "drive": 0}
    for _ in range(3):
        wins["drive"] += 1
        ledger.note(dict(wins), 0.0)
    wins["memory"] += 1
    ledger.note(dict(wins), 1.0)
    once = ledger.earned("memory")
    wins["drive"] += 1
    ledger.note(dict(wins), 0.0)
    wins["memory"] += 1
    ledger.note(dict(wins), 1.0)
    twice = ledger.earned("memory")
    assert 0.0 < once < twice < 1.0


def test_a_payoff_every_turn_brings_is_not_something_a_source_earned():
    """The baseline is what her turns usually bring; a source that only matches it earns nothing."""
    ledger = CreditLedger()
    for turn in range(1, 30):
        ledger.note({"memory": turn}, 0.6)
    assert abs(ledger.earned("memory")) < 0.05


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


def test_an_alarm_is_not_blamed_for_the_bad_days_it_reports():
    """It wins every turn that starts low, and those turns go as low turns go."""
    ledger = CreditLedger()
    wins = {"alarm": 0, "memory": 0}
    for _ in range(40):
        wins["alarm"] += 1
        ledger.note(dict(wins), -0.5, arrived_with=-0.5)
        wins["memory"] += 1
        ledger.note(dict(wins), 0.5, arrived_with=0.5)
    assert abs(ledger.earned("alarm")) < 0.1
    assert abs(ledger.earned("memory")) < 0.1


def test_a_source_whose_bad_days_go_worse_than_other_bad_days_loses():
    ledger = CreditLedger()
    wins = {"alarm": 0, "harm": 0, "memory": 0}
    for turn in range(60):
        if turn % 3 == 0:
            wins["harm"] += 1
            ledger.note(dict(wins), -1.0, arrived_with=-0.5)
        elif turn % 3 == 1:
            wins["alarm"] += 1
            ledger.note(dict(wins), -0.4, arrived_with=-0.5)
        else:
            wins["memory"] += 1
            ledger.note(dict(wins), 0.5, arrived_with=0.5)
    assert ledger.earned("harm") < ledger.earned("alarm")
    assert ledger.earned("harm") < 0.0


def test_the_baseline_is_what_turns_from_that_start_have_brought():
    ledger = CreditLedger()
    for turn in range(1, 21):
        ledger.note({"memory": turn}, 0.2 * (1 if turn % 2 else -1), arrived_with=0.5 if turn % 2 else -0.5)
    assert ledger.baseline(0.5) > 0.1
    assert ledger.baseline(-0.5) < -0.1
