"""A reading that cannot be taken in time is no opinion, never a failure of its caller.

LIVE 2026-09-26: a memory-pressure sysctl, started as a process, ran past its
two seconds on a loaded machine. The gateway's timeout was not a
``TimeoutError``, so it got past the handler meant for it, and the router took
it for her own model failing: "Endpoint Cortex raised exception: ... timed out
after 2.0 seconds". A 1.5B stand-in then answered a question about her.
"""

from __future__ import annotations

import pickle
import subprocess
import sys

import pytest

from core.runtime.subprocess_gateway import WorkBoundExpired
from core.utils import memory_monitor


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(memory_monitor, "_KERNEL_PRESSURE_CACHE", (0.0, memory_monitor.MEMORY_PRESSURE_UNKNOWN))


def test_the_gateways_timeout_is_a_timeout():
    expired = WorkBoundExpired(["sysctl", "-n", "x"], 2.0, reason="wall clock")
    assert isinstance(expired, TimeoutError)
    assert isinstance(expired, subprocess.TimeoutExpired)
    again = pickle.loads(pickle.dumps(expired))
    assert (again.cmd, again.timeout, again.reason) == (["sysctl", "-n", "x"], 2.0, "wall clock")


@pytest.mark.skipif(sys.platform != "darwin", reason="the kernel's pressure level is a Darwin sysctl")
def test_the_kernel_is_asked_without_starting_a_process(monkeypatch):
    def no_process(*_args, **_kwargs):
        raise AssertionError("a process was started to read a kernel value")

    from core.runtime import subprocess_gateway

    monkeypatch.setattr(subprocess_gateway.get_subprocess_gateway(), "run", no_process)
    assert memory_monitor.kernel_memory_pressure_level() in {
        memory_monitor.MEMORY_PRESSURE_NORMAL,
        memory_monitor.MEMORY_PRESSURE_WARN,
        memory_monitor.MEMORY_PRESSURE_CRITICAL,
    }


def test_a_reading_that_ran_out_of_time_is_no_opinion(monkeypatch):
    monkeypatch.setattr(memory_monitor.sys, "platform", "darwin")
    monkeypatch.setattr(memory_monitor, "_kernel_int", lambda name: None)

    class Gateway:
        def run(self, *_args, **_kwargs):
            raise WorkBoundExpired(["sysctl"], 2.0, reason="wall clock")

    from core.runtime import subprocess_gateway

    monkeypatch.setattr(subprocess_gateway, "get_subprocess_gateway", lambda: Gateway())
    assert memory_monitor.kernel_memory_pressure_level() == memory_monitor.MEMORY_PRESSURE_UNKNOWN
