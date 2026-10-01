"""A run's declared host reaches the substrate's noise on the scale the machine's readings do.

The chaos engine's somatic noise has an amplitude of one half plus the variance
of what it reads. The machine's readings arrive as fractions; the declared host
arrived as percents and degrees with every flag and count beside them, and the
noise came out more than a thousand times larger than the live runtime's.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import core.consciousness.controlled_chaos as chaos
from core.subject.declared_entropy import HOST_READINGS, _declare_the_chaos


def _runtime() -> SimpleNamespace:
    hardware = {
        "cpu_usage": 87.5,
        "ram_usage": 61.0,
        "temperature": 91.9,
        "vram_usage": 88.2,
        "motor_cortex_actions": 5_000,
        "battery_available": True,
        "battery": None,
    }
    return SimpleNamespace(state=SimpleNamespace(soma=SimpleNamespace(hardware=hardware)), seed=23)


def test_the_declared_signals_are_fractions_of_the_held_host_alone() -> None:
    try:
        _declare_the_chaos(_runtime())
        signals = chaos._DECLARED_SIGNALS()
    finally:
        chaos.declare_the_machine(None, None)
    assert len(signals) == 3 + len(HOST_READINGS)
    assert all(0.0 <= value <= 1.0 for value in signals)


def test_declared_noise_is_as_loud_as_the_machines() -> None:
    engine = chaos.ChaosEngine(chaos.ChaosConfig(state_dim=64))
    machine = float(np.linalg.norm(engine._poll_hardware_state(64)))
    try:
        _declare_the_chaos(_runtime())
        declared = float(np.linalg.norm(engine._poll_hardware_state(64)))
    finally:
        chaos.declare_the_machine(None, None)
    # Both are one half plus the variance of a handful of fractions.
    assert 0.5 <= declared < 1.0
    assert 0.5 <= machine < 1.0


def test_giving_the_machine_back_takes_effect_on_the_next_step() -> None:
    engine = chaos.get_chaos_engine(chaos.ChaosConfig(state_dim=64))
    try:
        _declare_the_chaos(_runtime())
        engine.tick(0.1)
    finally:
        chaos.declare_the_machine(None, None)
    assert engine._last_somatic_poll == 0.0
