"""The kernel's memory verdict is about this machine, not a declared one.

The memory snapshot carries the percentages its observer reports and, beside
them, what macOS says about memory pressure. The inference gate and the
background deferral both let a kernel verdict of "normal" lift the percentage
limit and a verdict of "critical" close it. When a measurement run declares its
host, the percentages are that host's, and the kernel's verdict was still this
machine's, so the real machine's load reached a run built to keep it out.
"""

from __future__ import annotations

from types import SimpleNamespace

from core.brain.llm_health_router import BRAINSTEM_ENDPOINT, EndpointHealth, HealthAwareLLMRouter
from core.runtime.resource_observation import SimulatedResourceObserver
from core.utils import memory_monitor
from core.utils.memory_monitor import MEMORY_PRESSURE_UNKNOWN, get_memory_pressure_snapshot


def _kernel_says(monkeypatch, level: str) -> list[int]:
    asked: list[int] = []

    def kernel() -> str:
        asked.append(1)
        return level

    monkeypatch.setattr(memory_monitor, "kernel_memory_pressure_level", kernel)
    return asked


def test_a_declared_host_carries_no_kernel_verdict(monkeypatch):
    asked = _kernel_says(monkeypatch, "critical")
    snapshot = get_memory_pressure_snapshot(observer=SimulatedResourceObserver(), force_refresh=True)
    assert snapshot.host_observed is False
    assert snapshot.kernel_pressure_level == MEMORY_PRESSURE_UNKNOWN
    assert asked == []


def _deferral_with(monkeypatch, *, host_observed: bool) -> list[int]:
    asked = _kernel_says(monkeypatch, "normal")
    monkeypatch.setenv("AURA_DESKTOP_RESOURCE_GUARD", "1")
    snap = SimpleNamespace(
        pressure_pct=60.0,
        available_gb=28.0,
        process_rss_gb=1.0,
        process_rss_limit_gb=40.0,
        host_observed=host_observed,
    )
    monkeypatch.setattr("core.utils.memory_monitor.get_memory_pressure_snapshot", lambda: snap)
    ep = EndpointHealth(name=BRAINSTEM_ENDPOINT, url="local://brainstem", model="qwen-7b")
    HealthAwareLLMRouter._desktop_background_endpoint_deferral_reason(ep)
    return asked


def test_the_background_deferral_does_not_ask_the_kernel_about_a_declared_host(monkeypatch):
    assert _deferral_with(monkeypatch, host_observed=False) == []


def test_it_still_asks_about_this_machine(monkeypatch):
    assert _deferral_with(monkeypatch, host_observed=True) == [1]
