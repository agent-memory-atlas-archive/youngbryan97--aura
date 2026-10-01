"""What an immune rule may be about, and the plant it may act on.

Lifted out of `adaptive_immunity`, which re-exports `_live_rule_vocabulary`.

A rule acts on the plant it measures. Sensors and actuators each name their
plant: "observed" for the host and the runtime she runs in, "simulated" for
the shipping network in core/world/world_model.py. Every actuator a rule may
be evolved with today moves the simulated network, because evolution runs
each rule in cloned worlds and only a simulated action can run in a clone. So
until 30 September a rule could read her event loop and "treat" it by moving
cargo in the simulation, and score the move as a repair of something real.
Now a sensor is offered to the rule author only when some actuator acts on
its plant, and the executor refuses a rule whose actions change a plant its
conditions do not measure. A real immune-compatible actuator brings her
runtime sensors back into the vocabulary without a change here.
"""
from __future__ import annotations

import copy
import logging
import math
from typing import Any

logger = logging.getLogger("Aura.AdaptiveImmunity")


def sensor_plant(name: str) -> str:
    """"observed" or "simulated" for a registered sensor; observed if unknown."""
    from core.sensors.sensor_registry import get_sensor_registry

    sensor = get_sensor_registry().sensors.get(str(name))
    return str(getattr(sensor, "source", "observed"))


def actuator_plant(name: str) -> str:
    """"observed" or "simulated" for a registered actuator; observed if unknown."""
    from core.actuators.actuator_registry import get_actuator_registry

    actuator = get_actuator_registry().get_actuator(str(name))
    return str(getattr(actuator, "plant", "observed"))


def sensors_on_actuated_plants(sensors: list[str], actuators: list[str]) -> list[str]:
    """The sensors some actuator in ``actuators`` can act on; [] if unreadable."""
    try:
        plants = {actuator_plant(name) for name in actuators}
        return [name for name in sensors if sensor_plant(name) in plants]
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("Immune rule vocabulary: plants unreadable, offering no sensors: %s", exc)
        return []


def plant_contract(
    conditions: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Refuse a rule whose actions change a plant its conditions do not measure."""
    measured = {sensor_plant(str(c.get("sensor") or "")) for c in conditions}
    acted = {actuator_plant(str(a.get("actuator") or "")) for a in actions}
    if measured and acted and not acted <= measured:
        return (
            False,
            f"rule measures the {'/'.join(sorted(measured))} plant and acts on the "
            f"{'/'.join(sorted(acted))} one",
        )
    return True, ""


def _live_rule_vocabulary() -> dict[str, Any] | None:
    """Sensors and actuators that ACTUALLY exist in this runtime.

    CP126 956ba926: rule generation drew from a hardcoded maritime vocabulary
    — port_east_load, vessel_alpha_speed, reallocate_flow(Port_East,
    Port_West). The immune system exists to repair Aura's subsystems, so a
    learning lane that can only express opinions about a logistics toy was
    optimizing something unrelated to its purpose and reporting the result as
    repair fitness.

    Returns None when neither registry can be read, which is the honest answer
    and makes the caller refuse to author a rule rather than fall back to the
    toy.
    """
    sensors: list[str] = []
    sensor_values: dict[str, float] = {}
    actuators: list[str] = []
    action_templates: dict[str, dict[str, Any]] = {}
    try:
        from core.sensors.sensor_registry import get_sensor_registry

        readings = get_sensor_registry().read_all()
        sensors = sorted(str(name) for name in readings)
        for name, value in readings.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                sensor_values[str(name)] = number
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("Immune rule vocabulary: sensors unavailable: %s", exc)
    try:
        from core.actuators.actuator_registry import get_actuator_registry

        registry = get_actuator_registry()
        for name, actuator in registry.actuators.items():
            if not bool(getattr(actuator, "immune_rule_compatible", False)):
                continue
            if bool(getattr(actuator, "requires_authority", True)):
                continue
            params = actuator.immune_rule_seed_params()
            if not isinstance(params, dict) or not actuator.validate_params(params):
                continue
            normalized_name = str(name)
            actuators.append(normalized_name)
            action_templates[normalized_name] = copy.deepcopy(params)
        actuators.sort()
    except (ImportError, RuntimeError, AttributeError, TypeError, ValueError) as exc:
        logger.debug("Immune rule vocabulary: actuators unavailable: %s", exc)
    sensors = sensors_on_actuated_plants(sensors, actuators)
    sensor_values = {name: value for name, value in sensor_values.items() if name in sensors}
    if not sensors or not actuators:
        return None
    return {
        "sensors": sensors,
        "sensor_values": sensor_values,
        "actuators": actuators,
        "action_templates": action_templates,
    }
