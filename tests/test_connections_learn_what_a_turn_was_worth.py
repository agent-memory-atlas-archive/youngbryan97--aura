"""Her connections learn what a turn was worth.

The unified field and the liquid substrate keep a trace of what each connection
did over a turn, and the turn's worth, signed and dosed by its size against her
recent turns, decides whether the connection strengthens or weakens. For the
field's input weights that is a competition: each unit's total input strength
is held where it was born, so an organ active before payoffs takes a share of
the field from the organs that were not.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from core.affect import what_it_was_worth as worth
from core.affect.what_it_was_worth import Worth, dose, teach_connections
from core.consciousness.liquid_substrate import LiquidSubstrate, SubstrateConfig
from core.consciousness.stdp_learning import STDPLearningEngine
from core.consciousness.unified_field import UnifiedField
from core.container import ServiceContainer
from core.phases.learning_phase import LearningPhase
from core.state.aura_state import AuraState


@pytest.fixture(autouse=True)
def fresh():
    worth.reset_for_test()
    ServiceContainer.clear()
    yield
    worth.reset_for_test()
    ServiceContainer.clear()


def _field_driven_by_the_mesh(ticks: int = 200) -> UnifiedField:
    field = UnifiedField()
    rng = np.random.default_rng(3)
    pattern = rng.uniform(-1.0, 1.0, field.cfg.mesh_input_dim).astype(np.float32)
    for _ in range(ticks):
        field.receive_mesh(pattern)
        field._tick()
    return field


def test_an_organ_active_before_a_payoff_takes_a_share_of_the_field():
    field = _field_driven_by_the_mesh()
    before = field.input_shares()
    lesson = field.teach(1.0)
    assert lesson["taught"] and lesson["steps"] > 0
    after = lesson["shares"]
    assert after["mesh"] > before["mesh"]
    for quiet in ("chemistry", "binding", "interoception", "substrate"):
        assert after[quiet] < before[quiet]


def test_an_organ_active_before_a_detriment_gives_its_share_up():
    field = _field_driven_by_the_mesh()
    before = field.input_shares()
    after = field.teach(-1.0)["shares"]
    assert after["mesh"] < before["mesh"]


def test_each_unit_keeps_the_input_strength_it_was_born_with():
    field = _field_driven_by_the_mesh()
    born = field._input_row_norms.copy()
    field.teach(1.0)
    assert np.allclose(np.linalg.norm(field._W_input_batched, axis=1), born, rtol=1e-4)


def test_the_blocks_and_the_batched_matrix_stay_one_matrix():
    field = _field_driven_by_the_mesh()
    field.teach(0.7)
    joined = np.hstack([field.W_mesh, field.W_chem, field.W_bind, field.W_intero, field.W_substrate])
    assert np.array_equal(joined, field._W_input_batched)


def test_a_neutral_turn_changes_nothing_and_ends_the_traces():
    field = _field_driven_by_the_mesh()
    weights = field._W_input_batched.copy()
    recurrent = field.W_field.copy()
    lesson = field.teach(0.0)
    assert not lesson["taught"]
    assert np.array_equal(field._W_input_batched, weights)
    assert np.array_equal(field.W_field, recurrent)
    assert field._eligible_steps == 0 and not field._eligible_input.any()


def test_the_recurrent_weights_are_taught_too():
    field = _field_driven_by_the_mesh()
    recurrent = field.W_field.copy()
    field.teach(1.0)
    assert not np.array_equal(field.W_field, recurrent)


def test_a_dose_that_is_not_a_number_teaches_nothing():
    field = _field_driven_by_the_mesh()
    weights = field._W_input_batched.copy()
    assert not field.teach(float("nan"))["taught"]
    assert np.array_equal(field._W_input_batched, weights)


def _engine_with_traces(seed: int = 5) -> STDPLearningEngine:
    engine = STDPLearningEngine(n_neurons=16)
    rng = np.random.default_rng(seed)
    for step in range(6):
        engine.record_spikes(rng.uniform(-1.0, 1.0, 16), t=step * 50.0)
    return engine


def test_better_strengthens_the_eligible_synapses_and_worse_weakens_them():
    better = _engine_with_traces().deliver_worth(1.0)
    worse = _engine_with_traces().deliver_worth(-1.0)
    assert np.abs(better).sum() > 0.0
    assert np.allclose(better, -worse)


def test_the_substrate_plasticity_step_no_longer_locks_every_synapse():
    """The per-step reward read a field the free-energy state does not have.

    It was zero on every step, so no weight moved, and the zero deltas drove
    each synapse's uncertainty down until all of them were identity-locked:
    256 of 256 by step 600 on 16 neurons.
    """
    ServiceContainer.register_instance(
        "free_energy_engine", SimpleNamespace(current=SimpleNamespace(surprise=0.3))
    )
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=16))
    rng = np.random.default_rng(7)
    for _ in range(700):
        substrate.x = rng.uniform(-1.0, 1.0, 16).astype(substrate.x.dtype)
        substrate._apply_plasticity_sync()
    engine = substrate._stdp_engine()
    assert int(engine._mesu_locked.sum()) == 0
    before = substrate.W.copy()
    lesson = substrate.teach(1.0)
    assert lesson["moved"] > 0.0
    assert not np.array_equal(substrate.W, before)


def test_the_dose_is_the_sign_times_the_size():
    assert dose(Worth(worth=2.0, size=0.8, measured=True)) == pytest.approx(0.8)
    assert dose(Worth(worth=-2.0, size=0.8, measured=True)) == pytest.approx(-0.8)
    assert dose(Worth(worth=0.0, size=0.8, measured=True)) == 0.0
    assert dose(Worth(worth=2.0, size=0.8, measured=False)) == 0.0


class _Organ:
    def __init__(self) -> None:
        self.doses: list[float] = []

    def teach(self, amount: float) -> dict[str, float]:
        self.doses.append(amount)
        return {"modulator": amount}


def test_every_taught_organ_is_told_even_on_a_neutral_turn():
    field, substrate = _Organ(), _Organ()
    ServiceContainer.register_instance("unified_field", field)
    ServiceContainer.register_instance("liquid_substrate", substrate)
    lessons = teach_connections(Worth())
    assert set(lessons) == {"unified_field", "liquid_substrate"}
    assert field.doses == [0.0] and substrate.doses == [0.0]


def test_the_learning_phase_teaches_the_organs_at_the_end_of_a_turn():
    field = _Organ()
    ServiceContainer.register_instance("unified_field", field)
    phase = LearningPhase(SimpleNamespace())
    state = AuraState()
    state.cognition.last_response = ""
    asyncio.run(phase.execute(state, objective="a turn"))
    assert field.doses == [0.0]
    assert state.response_modifiers["worth"]["taught"] == ["unified_field"]


def test_the_substrate_still_holds_its_weights_in_bounds_every_step():
    """The regulation used to ride on the dead reward delivery; it runs on its own now."""
    substrate = LiquidSubstrate(config=SubstrateConfig(neuron_count=16))
    substrate.W = np.full((16, 16), 2.0, dtype=substrate.W.dtype)
    np.fill_diagonal(substrate.W, 0.0)
    substrate.x = np.ones(16, dtype=substrate.x.dtype)
    before = float(np.linalg.norm(substrate.W, ord=2))
    substrate._apply_plasticity_sync()
    # Capped at 3.0, then nudged by the homeostatic step that follows the cap.
    assert before > 20.0
    assert np.linalg.norm(substrate.W, ord=2) < 3.0 * 1.01


class _Workspace:
    def __init__(self) -> None:
        self.wins = {"memory": 0}

    def wins_by_source(self) -> dict[str, int]:
        return dict(self.wins)


def test_the_sources_that_won_the_turn_are_credited_with_it():
    from core.affect import what_winning_earned

    what_winning_earned.reset_for_test()
    workspace = _Workspace()
    ServiceContainer.register_instance("global_workspace", workspace)
    teach_connections(Worth())
    workspace.wins["memory"] = 4
    lessons = teach_connections(Worth(worth=1.0, size=0.9, measured=True))
    assert lessons["global_workspace"] == {"memory": 1.0}
    assert what_winning_earned.get_credit_ledger().earned("memory") > 0.0
    what_winning_earned.reset_for_test()
