"""What she predicts and repairs is what was measured, and a simulation says so.

Until 30 September the sensor registry was one dictionary in which seven
readings came from the simulated shipping network in core/world/world_model.py
and fourteen runtime sensors were declared and never written. The closed loop
weighted that dictionary at 0.85 of her free energy as "physical" telemetry,
so her surprise about the world was the simulation compared with itself; an
immune rule could read her event loop and "treat" it by moving simulated
cargo; and the feed told the move as "Moved 161.8 of load from port east to
port west", as if cargo had moved.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from core.sensors import sensor_registry as sensors_module
from core.sensors.sensor_registry import SensorRegistry

ROOT = Path(__file__).resolve().parents[1]


class _Held(SensorRegistry):
    """A registry whose readings the test sets, with nothing synced over them."""

    def sync_from_world_model(self) -> None:
        return None


def test_every_sensor_says_where_its_readings_come_from() -> None:
    registry = SensorRegistry()
    provenance = registry.provenance()
    assert provenance["port_east_load"] == "unread"
    assert registry.sensors["port_east_load"].source == "simulated"
    assert registry.sensors["runtime_event_loop_lag"].source == "observed"
    registry.record_reading("port_east_load", 900.0)
    assert registry.provenance()["port_east_load"] == "simulated"


def test_an_unread_sensor_is_left_out_rather_than_read_as_zero() -> None:
    registry = SensorRegistry()
    registry.record_reading("system_cpu_usage", 12.0)
    registry.record_reading("port_east_load", 900.0)
    assert registry.read_observed() == {"system_cpu_usage": 12.0}


def test_the_host_is_read_on_every_sync() -> None:
    registry = SensorRegistry()
    registry.sync_observed()
    observed = registry.read_observed()
    for sensor in ("system_cpu_usage", "runtime_memory_pressure", "runtime_degradation_rate"):
        assert sensor in observed, observed
    assert 0.0 < observed["runtime_memory_pressure"] < 1.0


def _free_energy(registry: SensorRegistry, expectations: dict[str, float], monkeypatch) -> float:
    from core.consciousness.closed_loop import SelfPredictiveCore

    monkeypatch.setattr(sensors_module, "get_sensor_registry", lambda: registry)
    core = SelfPredictiveCore(neuron_count=64)
    core.predict(np.zeros(64, dtype=np.float32))
    cycle = core.observe_and_update(np.zeros(64, dtype=np.float32), simulated_expectations=expectations)
    assert cycle is not None
    return cycle.free_energy


def test_a_simulated_upheaval_is_no_surprise_about_her_world(monkeypatch) -> None:
    registry = _Held()
    for value in (10.0, 12.0, 11.0):
        registry.record_reading("system_cpu_usage", value)
    for value in (0.0, 900.0):
        registry.record_reading("port_east_load", value)
    energy = _free_energy(registry, {"system_cpu_usage": 11.0, "port_east_load": 0.0}, monkeypatch)
    assert energy == pytest.approx(0.0)


def test_a_measured_change_is_surprise_in_its_own_units(monkeypatch) -> None:
    registry = _Held()
    for value in (10.0, 12.0, 11.0):
        registry.record_reading("system_cpu_usage", value)
    energy = _free_energy(registry, {"system_cpu_usage": 5.0}, monkeypatch)
    assert energy > 1.0


def test_a_rule_may_not_treat_a_real_condition_with_a_simulated_remedy() -> None:
    from core.adaptation.immune_executor import ImmuneHeuristicExecutor

    ok, message = ImmuneHeuristicExecutor._rule_plant_contract(
        [{"sensor": "runtime_event_loop_lag", "operator": ">", "value": 0.5}],
        [{"actuator": "reallocate_flow", "params": {}}],
    )
    assert not ok
    assert "observed" in message and "simulated" in message
    ok, _ = ImmuneHeuristicExecutor._rule_plant_contract(
        [{"sensor": "port_east_load", "operator": ">", "value": 750.0}],
        [{"actuator": "reallocate_flow", "params": {}}],
    )
    assert ok


def test_a_move_on_the_simulation_says_it_was_simulated() -> None:
    from core.actuators.actuator_registry import get_actuator_registry

    actuator = get_actuator_registry().get_actuator("reallocate_flow")
    assert actuator.plant == "simulated"
    assert get_actuator_registry().get_actuator("reroute_vessel").plant == "simulated"
    source = (ROOT / "core/actuators/actuator_registry.py").read_text("utf-8")
    assert "transferred %s from %s to %s in the simulated plant" in source


def test_the_feed_tells_the_move_as_practice_on_a_simulation() -> None:
    script = (ROOT / "interface/static/aura.js").read_text("utf-8")
    rule = re.search(r"\[/\^Executed Actuator: \(\\w\+\) transferred.*?\n.*?\],", script, re.S)
    assert rule is not None
    assert "in the simulated plant" in rule.group(0)
    assert "shipping simulation" in rule.group(0)
