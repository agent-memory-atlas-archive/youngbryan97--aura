"""The shutdown verdict is an atomic write with an fsync, and it was written on
the event loop: every task still finishing waited on the disk for it, and
lockdep said so on every shutdown.

A whole graceful shutdown runs here with its parts made inert, and the verdict
writer notes the thread it ran on. It runs twice: with the loop's executor
still there, and with it already gone, which is how the last write of a real
shutdown finds it.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest


class _Clean:
    clean = True
    failed_phases: tuple = ()
    handler_failures: dict = {}


class _Coordinator:
    async def shutdown(self):
        return _Clean()


class _Container:
    async def shutdown(self):
        return {"clean": True}


async def _nothing(*_a, **_k):
    return None


@pytest.mark.parametrize("pool_gone", [False, True])
def test_the_verdict_is_published_off_the_loop(monkeypatch, pool_gone):
    from core.ops import graceful_shutdown as gs
    from core.runtime import executors

    written: list[tuple[int, dict]] = []

    def publish(**verdict):
        written.append((threading.get_ident(), verdict))

    monkeypatch.setattr(gs, "publish_shutdown_verdict", publish)
    # The latch is process-wide; this shutdown is the test's own.
    monkeypatch.setattr(gs, "request_shutdown", lambda *_a, **_k: None)
    monkeypatch.setattr(gs, "get_shutdown_coordinator", lambda: _Coordinator())
    monkeypatch.setattr(gs, "_arm_exit_stall_dump", lambda *_a, **_k: None)
    monkeypatch.setattr("core.container.get_container", lambda: _Container())
    monkeypatch.setattr("core.runtime.lifecycle_probe.hold_shutdown_probe_async", _nothing)
    monkeypatch.setattr(
        "core.runtime.runtime_hygiene.get_runtime_hygiene",
        lambda: SimpleNamespace(get_shutdown_report=lambda: {"swept": 0}),
    )
    monkeypatch.setattr(gs.GracefulShutdown, "_hooks", [])
    monkeypatch.setattr(gs.GracefulShutdown, "_is_shutting_down", False)
    monkeypatch.setattr(gs.GracefulShutdown, "_shutdown_event", None)
    monkeypatch.setattr(gs.GracefulShutdown, "_shutdown_owner_task", None)
    if pool_gone:

        async def gone(*_a, **_k):
            raise RuntimeError("cannot schedule new futures after shutdown")

        monkeypatch.setattr(executors.asyncio, "to_thread", gone)

    async def shut_down() -> int:
        await gs.GracefulShutdown.trigger_shutdown()
        return threading.get_ident()

    loop_thread = asyncio.run(shut_down())

    assert len(written) == 1, "the shutdown wrote no verdict"
    wrote_on, verdict = written[0]
    assert wrote_on != loop_thread, "the verdict was fsynced on the loop"
    assert verdict["final"] is True
    assert verdict["stage"] == "graceful_shutdown_complete"
    assert verdict["runtime_hygiene_report"] == {"swept": 0}
