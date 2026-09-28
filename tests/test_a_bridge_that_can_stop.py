"""The recovery bridge's worker leaves when it is stopped.

Its drain loop said it was "bounded by the started flag". Nothing cleared
the flag, and clearing it could not wake a thread blocked in `queue.get()`,
so every bridge built in a process kept a worker until the process ended.
"""
from __future__ import annotations

from core.resilience.recovery_bridge import RecoveryBridge


def test_stop_ends_the_worker(monkeypatch):
    monkeypatch.setenv("AURA_RECOVERY_BRIDGE", "1")
    bridge = RecoveryBridge()
    assert bridge.start() is True
    worker = bridge._worker
    assert worker is not None and worker.is_alive()

    assert bridge.stop(timeout_s=5.0) is True
    assert not worker.is_alive(), "the worker was still blocked in queue.get()"
    assert bridge.status()["started"] is False


def test_stop_is_safe_twice_and_before_start(monkeypatch):
    monkeypatch.setenv("AURA_RECOVERY_BRIDGE", "1")
    bridge = RecoveryBridge()
    assert bridge.stop() is True
    bridge.start()
    assert bridge.stop() is True
    assert bridge.stop() is True


def test_a_stopped_bridge_can_start_again(monkeypatch):
    monkeypatch.setenv("AURA_RECOVERY_BRIDGE", "1")
    bridge = RecoveryBridge()
    bridge.start()
    first = bridge._worker
    bridge.stop()
    assert bridge.start() is True
    assert bridge._worker is not first and bridge._worker.is_alive()
    bridge.stop()
