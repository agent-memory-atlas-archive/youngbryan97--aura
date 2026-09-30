"""A condition that lasts is a warning when it starts or worsens, not on every look.

R08, from the live log of 24-29 September: "HIGH MEMORY PRESSURE" 574 times on
26 September, "RAM CRITICAL ... Strike N" 210, and "Sustained distress" up to 46
a day, each the same condition warned again on every pass of its loop. Each
now warns when it starts, the monitor and the guard again when the reading
climbs a step, and says at info when it is over.
"""
from __future__ import annotations

import asyncio
import logging
import types

import pytest

from core.utils.standing_condition import StandingCondition


def _warnings(caplog: pytest.LogCaptureFixture, text: str) -> int:
    return sum(1 for r in caplog.records if r.levelno == logging.WARNING and text in r.getMessage())


def test_the_first_look_warns_and_the_rest_do_not(caplog: pytest.LogCaptureFixture) -> None:
    log = logging.getLogger("test.standing")
    condition = StandingCondition(log, ended="over")
    with caplog.at_level(logging.DEBUG, logger="test.standing"):
        said = [condition.report(True, "hot %s", n) for n in range(10)]
    assert said == [True] + [False] * 9
    assert _warnings(caplog, "hot") == 1
    assert sum(1 for r in caplog.records if r.levelno == logging.DEBUG) == 9


def test_worse_by_a_step_is_said_again(caplog: pytest.LogCaptureFixture) -> None:
    condition = StandingCondition(logging.getLogger("test.standing"), ended="over", step=5.0)
    with caplog.at_level(logging.DEBUG, logger="test.standing"):
        for reading in (86, 87, 89, 91, 92, 96, 97):
            condition.report(True, "pressure %s", reading, level=float(reading))
    assert _warnings(caplog, "pressure") == 3  # 86, 91, 96


def test_the_end_is_said_once_at_info_and_a_return_warns_again(caplog: pytest.LogCaptureFixture) -> None:
    condition = StandingCondition(logging.getLogger("test.standing"), ended="it is over")
    with caplog.at_level(logging.DEBUG, logger="test.standing"):
        condition.report(True, "on")
        condition.report(False, "on")
        condition.report(False, "on")
        condition.report(True, "on")
    assert sum(1 for r in caplog.records if r.levelno == logging.INFO and r.getMessage() == "it is over") == 1
    assert _warnings(caplog, "on") == 2


def test_sustained_distress_is_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    from core.consciousness.hedonic_gradient import HedoniGradientEngine

    gradient = HedoniGradientEngine()
    with caplog.at_level(logging.DEBUG, logger="Aura.HedoniGradient"):
        for _ in range(40):
            gradient.update(valence=-1.0, arousal=0.05, curiosity=0.0, energy=0.0)
    assert _warnings(caplog, "Sustained distress") == 1


def test_the_memory_monitor_warns_on_the_crossing_and_the_climb(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.utils.memory_monitor import AppleSiliconMemoryMonitor

    monitor = AppleSiliconMemoryMonitor(interval=0.0, threshold=50)
    readings = iter([55, 55, 56, 55, 61, 62, 40, 40])

    def sample() -> int:
        try:
            return next(readings)
        except StopIteration:
            monitor.is_running = False
            return 40

    monkeypatch.setattr(monitor, "_get_pressure_sysctl", sample)
    monitor.is_running = True
    with caplog.at_level(logging.DEBUG, logger="Aura.MemoryMonitor"):
        asyncio.run(monitor._monitor_loop())
    assert _warnings(caplog, "HIGH MEMORY PRESSURE") == 2  # 55, then 61
    assert any("back under its threshold" in r.getMessage() for r in caplog.records)


def test_the_memory_guard_warns_on_the_first_strike_and_the_climb(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    from core.guardians import memory_guard
    from core.utils import memory_monitor

    guard = memory_guard.MemoryGuard(threshold_percent=82.0)
    readings = iter([86, 86, 87, 86, 90, 70])

    def sample(_self: object) -> int:
        try:
            return next(readings)
        except StopIteration:
            guard._running = False
            return 70

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(memory_monitor.AppleSiliconMemoryMonitor, "_get_pressure_sysctl", sample)
    # Only this module's own name for asyncio: the process's sleep is left alone.
    monkeypatch.setattr(memory_guard, "asyncio", types.SimpleNamespace(sleep=no_wait, Task=asyncio.Task))
    monkeypatch.setattr(memory_guard, "get_runtime_service", lambda *_a, **_k: None)
    guard._running = True
    with caplog.at_level(logging.DEBUG, logger="Aura.MemoryGuard"):
        asyncio.run(guard._watch_loop())
    assert _warnings(caplog, "RAM CRITICAL") == 2  # 86, then 90
