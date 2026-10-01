"""The substrate's chaos is declared by a subject-core run, and carried by its fork.

The controlled chaos engine perturbs the liquid substrate on every step. Its
somatic noise hashed the machine's uptime, its own process's thread count and
memory, and the CPU's temperature, seeded from the process id and its start
time, and the engine is made after the fork is calibrated, so nothing carried
its history: one turn after another in a different condition differed in the
substrate's field, and two processes drove C with different noise.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.consciousness import controlled_chaos as chaos
from core.subject.driver import install_declared_host, release_declared_host

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _undeclared():
    yield
    chaos.declare_the_machine(None, None)


def test_a_declared_machine_is_all_the_somatic_noise_reads(monkeypatch) -> None:
    def refuse(self):
        raise AssertionError("a declared engine read the machine")

    monkeypatch.setattr(chaos.ChaosEngine, "_machine_signals", refuse)
    chaos.declare_the_machine(lambda: [0.25, 0.5], b"seed" * 8)
    one = chaos.ChaosEngine(chaos.ChaosConfig())
    two = chaos.ChaosEngine(chaos.ChaosConfig())
    dim = one.config.state_dim
    np.testing.assert_array_equal(one._poll_hardware_state(dim), two._poll_hardware_state(dim))


def test_an_undeclared_engine_still_reads_its_machine(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(chaos.ChaosEngine, "_machine_signals", lambda self: called.append(1) or [0.5])
    engine = chaos.ChaosEngine(chaos.ChaosConfig())
    engine._poll_hardware_state(engine.config.state_dim)
    assert called


def test_the_fork_carries_each_engines_history() -> None:
    from core.subject.snapshot import _chaos_history, _restore_chaos

    engine = chaos.get_chaos_engine()
    engine._tick_count = 7
    engine._prev_chem_levels = np.arange(3.0)
    saved = _chaos_history()
    engine._tick_count = 99
    engine._prev_chem_levels = None
    _restore_chaos(saved)
    assert engine._tick_count == 7
    np.testing.assert_array_equal(engine._prev_chem_levels, np.arange(3.0))


def test_the_declared_host_declares_the_chaos_and_gives_it_back() -> None:
    runtime = SimpleNamespace(state=SimpleNamespace(soma=SimpleNamespace(hardware={"cpu_usage": 30.0})),
                              declared_host=None, seed=23)
    install_declared_host(runtime)
    try:
        assert chaos._DECLARED_SIGNALS is not None
        assert chaos._DECLARED_SEED is not None
        # A percent, read on the machine's own scale.
        assert 0.3 in chaos._DECLARED_SIGNALS()
    finally:
        release_declared_host(runtime)
    assert chaos._DECLARED_SIGNALS is None
