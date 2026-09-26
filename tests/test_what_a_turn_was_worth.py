"""What a turn was worth to her, against what she had come to expect.

core/affect/what_it_was_worth.py reads each turn's payoff on her own channels
and keeps what each has come to pay. A payoff better than expected is a
positive error, worse is negative, as expected is none; the signed sum reaches
her chemistry as a burst or a dip.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from core.affect import what_it_was_worth as worth
from core.affect.what_it_was_worth import CHANNELS, WorthLedger, broadcast, read_turn
from core.consciousness.neurochemical_system import NeurochemicalSystem
from core.container import ServiceContainer
from core.phases.learning_phase import LearningPhase
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def fresh_ledger():
    worth.reset_for_test()
    ServiceContainer.clear()
    yield
    worth.reset_for_test()
    ServiceContainer.clear()


def _warm(ledger: WorthLedger, channel: str, sizes: list[float]) -> None:
    """Give a channel enough changes to have a spread and an expectation."""
    for value in sizes:
        ledger.note({channel: value})


def test_a_payoff_repeated_turn_after_turn_stops_being_news():
    ledger = WorthLedger()
    readings = [ledger.note({"warmth": 0.4}) for _ in range(6)]
    measured = [reading for reading in readings if reading.measured]
    assert measured, "the channel never had enough changes to be read"
    assert measured[0].error["warmth"] > 0.0
    assert measured[-1].error["warmth"] == pytest.approx(0.0)
    assert measured[-1].worth == pytest.approx(0.0)


def test_better_than_expected_is_positive_and_worse_is_negative():
    ledger = WorthLedger()
    _warm(ledger, "satisfaction", [0.1, 0.1, 0.1, 0.1, 0.1])
    better = ledger.note({"satisfaction": 0.5})
    assert better.worth > 0.0
    _warm(ledger, "satisfaction", [0.1, 0.1, 0.1])
    worse = ledger.note({"satisfaction": -0.5})
    assert worse.worth < 0.0
    assert "worse" in worse.why and "better" in better.why


def test_an_expected_payoff_that_does_not_come_is_a_loss():
    ledger = WorthLedger()
    _warm(ledger, "accomplishment", [1.0, 1.0, 1.0, 1.0, 1.0])
    # The channel has come to pay about one of its spreads a turn, so a turn
    # that pays nothing is worse than expected.
    nothing = ledger.note({"accomplishment": 0.0})
    assert nothing.measured and nothing.worth < 0.0


def test_each_channel_is_read_in_its_own_spread():
    ledger = WorthLedger()
    for _ in range(4):
        ledger.note({"warmth": 0.001, "excitement": 10.0})
    reading = ledger.note({"warmth": 0.002, "excitement": 20.0})
    # Both doubled against their own spread, so each pays the same.
    assert reading.payoff["warmth"] == pytest.approx(reading.payoff["excitement"])


def test_the_size_ranks_this_turn_against_her_recent_turns():
    ledger = WorthLedger()
    _warm(ledger, "warmth", [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1])
    ordinary = ledger.note({"warmth": 0.15})
    largest = ledger.note({"warmth": 3.0})
    assert 0.0 <= ordinary.size < largest.size <= 1.0


def test_no_change_on_a_channel_she_has_not_heard_from_is_unread():
    reading = WorthLedger().note({})
    assert not reading.measured
    assert reading.worth == 0.0


def test_a_reading_that_is_not_a_number_is_no_change():
    ledger = WorthLedger()
    _warm(ledger, "wonder", [1.0, -1.0, 1.0])
    reading = ledger.note({"wonder": "not a number"})
    assert reading.payoff.get("wonder", 0.0) == pytest.approx(0.0)


def _state(valence: float, arousal: float) -> AuraState:
    state = AuraState()
    state.affect.valence = valence
    state.affect.arousal = arousal
    return state


def test_the_four_affect_channels_are_the_quadrants_of_valence_by_arousal():
    levels = worth._levels(_state(0.8, 0.9))
    assert levels["excitement"] > levels["peace"] > 0.0
    assert levels["ease"] == 0.0 and levels["spirit"] == 0.0
    levels = worth._levels(_state(0.8, 0.1))
    assert levels["peace"] > levels["excitement"] > 0.0
    levels = worth._levels(_state(-0.8, 0.9))
    assert levels["ease"] < levels["spirit"] < 0.0
    assert levels["excitement"] == 0.0 and levels["peace"] == 0.0
    levels = worth._levels(_state(-0.8, 0.1))
    assert levels["spirit"] < levels["ease"] < 0.0


def test_a_turn_is_read_against_the_end_of_the_one_before():
    ledger = WorthLedger()
    for valence in (0.1, 0.2, 0.1, 0.2, 0.1):
        read_turn(_state(valence, 0.8), ledger)
    before = ledger.read()
    lifted = read_turn(_state(0.9, 0.8), ledger)
    assert lifted.measured
    assert lifted.error["excitement"] > 0.0
    assert lifted.turns == before.turns + 1


def test_a_finished_goal_pays_once_and_a_failed_one_costs():
    ledger = WorthLedger()
    state = _state(0.0, 0.5)
    state.cognition.active_goals = [{"id": "g1", "goal": "write it", "status": "done"}]
    first = worth._goal_outcomes(state, ledger._outcomes_seen)
    again = worth._goal_outcomes(state, ledger._outcomes_seen)
    state.cognition.active_goals.append({"id": "g2", "goal": "run it", "status": "failed"})
    failed = worth._goal_outcomes(state, ledger._outcomes_seen)
    assert (first, again, failed) == (1.0, 0.0, -1.0)


def test_needs_being_met_is_satisfaction():
    state = _state(0.0, 0.5)
    state.motivation.budgets = {"energy": {"level": 40.0, "capacity": 100.0}}
    hungry = worth._levels(state)["satisfaction"]
    state.motivation.budgets = {"energy": {"level": 90.0, "capacity": 100.0}}
    fed = worth._levels(state)["satisfaction"]
    assert fed > hungry


def test_wonder_is_her_self_model_surprised_and_signed_by_how_she_felt():
    class _SelfModel:
        count = 0

        def get_snapshot(self):
            return {"surprise_count": self.count}

    model = _SelfModel()
    ServiceContainer.register_instance("self_prediction", model)
    ledger = WorthLedger()
    read_turn(_state(0.5, 0.5), ledger)
    model.count = 1
    read_turn(_state(0.5, 0.5), ledger)
    assert list(ledger._changes["wonder"]) == [1.0]
    model.count = 2
    read_turn(_state(-0.5, 0.5), ledger)
    assert list(ledger._changes["wonder"]) == [1.0, -1.0]


class _Chemistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float]] = []

    def on_reward(self, magnitude: float) -> None:
        self.calls.append(("reward", magnitude))

    def on_disappointment(self, magnitude: float) -> None:
        self.calls.append(("disappointment", magnitude))


def test_better_is_a_burst_and_worse_is_a_dip_dosed_by_its_size():
    chemistry = _Chemistry()
    ServiceContainer.register_instance("neurochemical_system", chemistry)
    ledger = WorthLedger()
    _warm(ledger, "warmth", [0.1, 0.2, 0.1, 0.2, 0.1, 0.2])
    better = ledger.note({"warmth": 2.0})
    assert broadcast(better) == "burst"
    _warm(ledger, "warmth", [0.1, 0.2, 0.1])
    worse = ledger.note({"warmth": -2.0})
    assert broadcast(worse) == "dip"
    assert chemistry.calls == [("reward", better.size), ("disappointment", worse.size)]


def test_as_expected_sends_nothing():
    chemistry = _Chemistry()
    ServiceContainer.register_instance("neurochemical_system", chemistry)
    ledger = WorthLedger()
    readings = [ledger.note({"warmth": 0.4}) for _ in range(6)]
    assert broadcast(readings[-1]) == "none"
    assert chemistry.calls == []


def test_the_dip_lowers_dopamine_and_is_capped_per_call():
    chemistry = NeurochemicalSystem()
    dopamine = chemistry.chemicals["dopamine"]
    before = dopamine.tonic_level
    chemistry.on_disappointment(1.0)
    fell = before - dopamine.tonic_level
    assert 0.0 < fell <= 0.08 + 1e-9


def test_every_channel_is_read_from_her_state():
    ledger = WorthLedger()
    state = _state(0.3, 0.6)
    read_turn(state, ledger)
    assert set(ledger._last_levels) >= {"satisfaction", "excitement", "peace", "ease", "spirit", "wonder_count"}
    assert set(CHANNELS) == {
        "satisfaction", "accomplishment", "warmth", "excitement", "peace", "ease", "spirit", "wonder",
    }


def test_the_learning_phase_reads_the_turn_even_when_there_was_no_reply():
    chemistry = _Chemistry()
    ServiceContainer.register_instance("neurochemical_system", chemistry)
    phase = LearningPhase(SimpleNamespace())
    state = _state(0.2, 0.5)
    state.cognition.last_response = ""
    asyncio.run(phase.execute(state, objective="nothing said"))
    recorded = state.response_modifiers["worth"]
    assert recorded["sent"] in {"none", "burst", "dip"}
    assert recorded["turns"] == 1
    assert worth.get_worth_ledger().turns == 1


def test_an_affect_that_is_not_a_number_leaves_the_affect_channels_unread():
    levels = worth._levels(_state(float("nan"), 0.5))
    assert not {"excitement", "peace", "ease", "spirit"} & set(levels)


def test_the_ablation_switch_takes_the_whole_layer_out(monkeypatch):
    """AURA_DISABLE_PAYOFF, read on the call: no turn read, nothing sent, nothing taught."""
    chemistry = _Chemistry()
    ServiceContainer.register_instance("neurochemical_system", chemistry)
    monkeypatch.setenv(worth.DISABLE_ENV, "1")
    phase = LearningPhase(SimpleNamespace())
    state = _state(0.2, 0.5)
    state.cognition.last_response = ""
    asyncio.run(phase.execute(state, objective="switched off"))
    assert "worth" not in state.response_modifiers
    assert worth.get_worth_ledger().turns == 0
    assert chemistry.calls == []
    monkeypatch.setenv(worth.DISABLE_ENV, "")
    assert not worth.disabled()
