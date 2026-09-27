"""What good turns leave her with, to work on.

core/soma/reserve.py: a turn better than she expected charges a reserve by its
dose times what an ordinary spending turn costs her, a worse one drains it, and
the reserve pays for her exertion before her energy does. Resting draws nothing.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.soma import reserve as reserve_module
from core.soma.reserve import ReserveLedger


@pytest.fixture(autouse=True)
def fresh():
    from core.affect import what_it_was_worth

    reserve_module.reset_for_test()
    what_it_was_worth.reset_for_test()
    yield
    reserve_module.reset_for_test()
    what_it_was_worth.reset_for_test()


def _spent(ledger: ReserveLedger, *amounts: float) -> None:
    for amount in amounts:
        ledger.draw(amount)


def test_nothing_is_charged_before_she_has_spent_anything():
    """No ordinary turn yet, so no dose her own life justifies."""
    ledger = ReserveLedger()
    assert ledger.charge(1.0) == 0.0
    assert ledger.held == 0.0


def test_her_best_turn_in_a_while_pays_back_one_ordinary_turn():
    ledger = ReserveLedger()
    _spent(ledger, 0.2, 0.4, 0.6)
    ledger.charge(1.0)
    assert ledger.held == pytest.approx(0.4)


def test_a_worse_turn_drains_it_and_never_below_nothing():
    ledger = ReserveLedger()
    _spent(ledger, 0.4, 0.4, 0.4)
    ledger.charge(0.5)
    ledger.charge(-1.0)
    assert ledger.held == 0.0


def test_the_reserve_pays_for_her_work_before_her_energy_does():
    ledger = ReserveLedger()
    _spent(ledger, 0.4, 0.4, 0.4)
    ledger.charge(1.0)
    covered = ledger.draw(0.3)
    assert covered == pytest.approx(0.3)
    assert ledger.held == pytest.approx(0.1)
    assert ledger.draw(0.3) == pytest.approx(0.1)
    assert ledger.draw(0.3) == 0.0


def test_resting_draws_nothing():
    ledger = ReserveLedger()
    _spent(ledger, 0.4, 0.4, 0.4)
    ledger.charge(1.0)
    assert ledger.draw(-0.2) == 0.0
    assert ledger.draw(0.0) == 0.0
    assert ledger.held == pytest.approx(0.4)


def test_it_holds_no_more_than_a_full_budget():
    ledger = ReserveLedger()
    ledger.draw(1.0, capacity=2.0)
    _spent(ledger, 1.5, 1.5)
    ledger.draw(0.0, capacity=2.0)
    for _ in range(10):
        ledger.charge(1.0)
    assert ledger.held <= 2.0 + 1e-9


def test_it_leaks_a_share_every_turn():
    ledger = ReserveLedger()
    _spent(ledger, 0.4, 0.4, 0.4)
    ledger.charge(1.0)
    held = ledger.held
    ledger.charge(0.0)
    assert ledger.held == pytest.approx(held * (1.0 - 1.0 / 256))


def test_a_dose_that_is_not_a_number_moves_nothing():
    ledger = ReserveLedger()
    _spent(ledger, 0.4, 0.4, 0.4)
    assert ledger.charge(float("nan")) == 0.0
    assert ledger.charge("lots") == 0.0


def test_energy_is_spent_less_while_the_reserve_holds(monkeypatch):
    from core.phases import motivation_update

    repo = SimpleNamespace(_current=SimpleNamespace(soma=SimpleNamespace(exertion=1.0)))
    monkeypatch.setattr(motivation_update, "get_runtime_service", lambda *_a, **_k: repo)
    spend = motivation_update.MotivationUpdatePhase._spend_energy
    plain = 100.0 - spend(100.0, 100.0, 60.0)
    ledger = reserve_module.get_reserve_ledger()
    _spent(ledger, plain, plain, plain)
    ledger.charge(1.0)
    covered_turn = 100.0 - spend(100.0, 100.0, 60.0)
    assert plain > 0.0
    assert covered_turn < plain


def test_the_learning_phase_charges_it_and_the_switch_takes_that_out(monkeypatch):
    from core.affect import what_it_was_worth
    from core.phases.learning_phase import LearningPhase
    from core.state.aura_state import AuraState

    ledger = reserve_module.get_reserve_ledger()
    _spent(ledger, 0.4, 0.4, 0.4)
    # A turn as good as her best in a while.
    better = what_it_was_worth.Worth(worth=2.0, size=1.0, measured=True, turns=9)
    monkeypatch.setattr(what_it_was_worth, "read_turn", lambda state, ledger=None: better)
    state = AuraState()
    state.cognition.last_response = ""
    asyncio.run(LearningPhase(SimpleNamespace()).execute(state, objective="a good turn"))
    assert ledger.held == pytest.approx(0.4)
    monkeypatch.setenv(what_it_was_worth.DISABLE_ENV, "1")
    asyncio.run(LearningPhase(SimpleNamespace()).execute(state, objective="switched off"))
    assert ledger.held == pytest.approx(0.4)


def test_the_core_reads_the_reserve_in_deliberation():
    from core.state.aura_state import AuraState
    from core.subject.state import Organs, read_core_state, schema

    ledger = ReserveLedger()
    _spent(ledger, 0.4, 0.4, 0.4)
    ledger.draw(0.0, capacity=4.0)
    ledger.charge(1.0)
    reading = read_core_state(AuraState.default(), organs=Organs(reserve=ledger))
    column = list(schema("D").features).index("energy_reserve")
    assert float(reading.values["D"][column]) == pytest.approx(0.1)
