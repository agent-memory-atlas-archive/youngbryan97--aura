"""The immune population must be able to express Aura's failures.

CP126 fae09510: behavioural rules are generated over whatever sensors exist,
and the only ones that existed described ports, vessels and a warehouse. So
every B cell Aura evolved was an opinion about maritime logistics, and the
adaptive population began with no code, service, queue, model, network,
memory or UI repair grammar. The vocabulary was the ceiling on what the
immune system could ever learn to repair.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.adaptation.adaptive_immunity import (
    _live_rule_vocabulary,
    _mutate_behavioral_rule,
)
from core.sensors.sensor_registry import SensorRegistry

_MARITIME = ("port_", "vessel_", "warehouse_")


@pytest.fixture()
def registry():
    return SensorRegistry()


def test_runtime_subsystems_have_sensors(registry):
    names = set(registry.read_all())
    runtime = {n for n in names if not n.startswith(_MARITIME)}

    assert len(runtime) >= 10


@pytest.mark.parametrize(
    "domain",
    ["runtime_", "model_", "memory_", "network_", "interface_", "storage_"],
)
def test_each_repair_domain_is_expressible(registry, domain):
    """A domain with no sensor is a domain no rule can be about."""
    assert any(name.startswith(domain) for name in registry.read_all())


def test_the_maritime_sensors_are_not_the_whole_vocabulary(registry):
    names = set(registry.read_all())
    maritime = {n for n in names if n.startswith(_MARITIME)}

    assert len(names - maritime) > len(maritime)


def test_seeded_rules_measure_only_a_plant_they_can_act_on():
    """30 September: a rule reading her event loop and "treating" it by moving
    simulated cargo scored a simulated move as a real repair. Every actuator a
    rule may evolve with acts on the simulated shipping network, so the seeded
    rules measure that network and nothing else."""
    vocabulary = _live_rule_vocabulary()
    assert vocabulary is not None

    rng = np.random.default_rng(0)
    sensors = set()
    for _ in range(60):
        rule = _mutate_behavioral_rule(None, rng)
        if rule:
            sensors.add(rule["conditions"][0]["sensor"])

    assert sensors and all(s.startswith(_MARITIME) for s in sensors), sensors


def test_a_real_immune_actuator_brings_her_runtime_sensors_back(monkeypatch):
    """The vocabulary is not maritime by design: it follows the actuators."""
    from core.actuators.actuator_registry import ActuatorResult, BaseActuator, get_actuator_registry

    class SheddingProbe(BaseActuator):
        immune_rule_compatible = True
        requires_authority = False
        plant = "observed"

        @property
        def name(self) -> str:
            return "test_observed_remedy"

        @property
        def description(self) -> str:
            return "A remedy on the observed plant, for this test."

        def validate_params(self, params):
            return True

        def execute(self, params):
            return ActuatorResult(True, "noted", {})

        def immune_rule_seed_params(self):
            return {"level": 1.0}

    registry = get_actuator_registry()
    monkeypatch.setitem(registry.actuators, "test_observed_remedy", SheddingProbe())
    vocabulary = _live_rule_vocabulary()
    assert vocabulary is not None
    assert any(not s.startswith(_MARITIME) for s in vocabulary["sensors"])


def test_the_population_is_not_one_rule_family():
    """Every initial B cell used to receive the same maritime flow rule."""
    rng = np.random.default_rng(3)
    rules = [_mutate_behavioral_rule(None, rng) for _ in range(20)]
    sensors = {r["conditions"][0]["sensor"] for r in rules if r}

    assert len(sensors) > 1


def test_a_declared_sensor_with_no_reading_is_still_declared(registry):
    """Declaring it is what lets a rule be ABOUT it, and what makes a missing
    reading visible rather than the subsystem invisible."""
    assert "runtime_event_loop_lag" in registry.read_all()
