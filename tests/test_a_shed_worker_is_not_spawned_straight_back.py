"""A worker shed for memory is not spawned back into the memory that shed it.

LIVE 2026-09-20, one uptime of the desktop runtime with the host at 80%:
the brainstem (Ternary Bonsai 2 27B, 8.6GB) was unloaded 176 times, each
``unloaded Ternary-Bonsai-2-27B-mlx-2bit to protect the foreground lane
(background_memory_pressure_shed)`` twenty seconds after the previous one.
The shed unloaded the worker and told nobody; the next background turn
spawned it again, paid the load, and the next shed unloaded it again.

The shed and the spawn already share a mechanism: the client's model-load
admission backoff, which a background spawn consults before it reaches for
the worker. The shed now records itself there as a denial, so the hold
grows while the sheds repeat and clears on the next admission that succeeds.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.inference_gate import InferenceGate

pytestmark = pytest.mark.unit


class _ShedClient:
    def __init__(self, path: str) -> None:
        self.model_path = path
        self.alive = True
        self.denials: list[tuple[str, str]] = []

    def is_alive(self) -> bool:
        return self.alive

    async def reboot_worker(self, reason: str = "", mark_failed: bool = False) -> None:
        self.alive = False

    def _note_model_load_admission_denial(self, reason: str, *, receipt_id: str) -> float:
        self.denials.append((reason, receipt_id))
        return 15.0


def _shed(gate: InferenceGate, client: _ShedClient) -> None:
    import core.brain.llm.mlx_client as mlx_client

    original = dict(getattr(mlx_client, "_CLIENTS", {}))
    mlx_client._CLIENTS.clear()
    mlx_client._CLIENTS[client.model_path] = client
    try:
        asyncio.run(
            gate._shed_background_workers_for_memory_pressure(
                force=True, reason="background_memory_pressure_shed"
            )
        )
    finally:
        mlx_client._CLIENTS.clear()
        mlx_client._CLIENTS.update(original)


@pytest.fixture()
def gate(monkeypatch: pytest.MonkeyPatch) -> InferenceGate:
    g = InferenceGate.__new__(InferenceGate)
    g._mlx_client = object()
    g._last_background_memory_shed_at = 0.0
    monkeypatch.setattr(
        "core.utils.memory_monitor.get_memory_pressure_snapshot",
        lambda: type("S", (), {"available_gb": 12.0})(),
        raising=False,
    )
    return g


def test_a_memory_shed_is_recorded_as_an_admission_denial(gate: InferenceGate) -> None:
    client = _ShedClient("/models/Ternary-Bonsai-2-27B-mlx-2bit")

    _shed(gate, client)

    assert not client.alive
    assert client.denials == [("memory_pressure_shed", "background_memory_pressure_shed")]


def test_the_hold_grows_while_the_sheds_repeat() -> None:
    """The client's own arithmetic: the memory branch, doubling per repeat."""
    from core.brain.llm.mlx_client import MLXLocalClient

    seconds = MLXLocalClient._model_load_admission_backoff_seconds
    first = seconds("memory_pressure_shed", 1)
    fourth = seconds("memory_pressure_shed", 4)
    assert first >= 15.0
    assert fourth > first
    assert seconds("memory_pressure_shed", 9) <= 300.0


@pytest.mark.parametrize(("background", "spawned"), [(True, False), (False, True)])
def test_a_background_spawn_honours_the_hold(
    monkeypatch: pytest.MonkeyPatch, background: bool, spawned: bool
) -> None:
    """The line the shed relies on: a background spawn asks the backoff first.

    A client that has just been shed is asked for its worker twice. The
    background ask stops at the hold without reaching admission; a person's
    turn goes through, because the hold is for background work.
    """
    import contextlib

    from core.brain.llm import mlx_client
    from core.brain.llm.mlx_client import MLXLocalClient

    client = MLXLocalClient("/models/Ternary-Bonsai-2-27B-mlx-2bit")
    client._note_model_load_admission_denial(
        "memory_pressure_shed", receipt_id="background_memory_pressure_shed"
    )
    reached: list[str] = []

    @contextlib.asynccontextmanager
    async def _admitted(_client, *, foreground_request):
        reached.append("admission")
        yield None

    @contextlib.asynccontextmanager
    async def _gate(**_kw):
        yield None

    async def _inner(**_kw):
        reached.append("spawn")
        return True

    async def _no_cooldown(**_kw):
        return None

    monkeypatch.setattr(mlx_client, "_foreground_owner_active", lambda: False)
    monkeypatch.setattr(mlx_client, "_background_deferral_active", lambda *_a, **_k: None)
    monkeypatch.setattr(mlx_client, "_model_load_admission_context", _admitted)
    monkeypatch.setattr(mlx_client, "_spawn_gate_context", _gate)
    monkeypatch.setattr(client, "_ensure_worker_alive_inner", _inner)
    monkeypatch.setattr(client, "_await_swap_cooldown", _no_cooldown)

    result = asyncio.run(
        client._ensure_worker_alive(
            request_is_background=background, foreground_request=not background
        )
    )
    assert result is spawned
    assert reached == (["admission", "spawn"] if spawned else [])


def test_a_worker_that_did_not_unload_is_not_held() -> None:
    """No hold for a shed that reclaimed nothing: the worker is still there."""
    import core.brain.llm.mlx_client as mlx_client

    class _Stays(_ShedClient):
        async def reboot_worker(self, reason: str = "", mark_failed: bool = False) -> None:
            pass

    g = InferenceGate.__new__(InferenceGate)
    g._mlx_client = object()
    g._last_background_memory_shed_at = 0.0
    client = _Stays("/models/stays-resident")
    original = dict(getattr(mlx_client, "_CLIENTS", {}))
    mlx_client._CLIENTS.clear()
    mlx_client._CLIENTS[client.model_path] = client
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "core.utils.memory_monitor.get_memory_pressure_snapshot",
                lambda: type("S", (), {"available_gb": 12.0})(),
                raising=False,
            )
            asyncio.run(
                g._shed_background_workers_for_memory_pressure(
                    force=True, reason="background_memory_pressure_shed"
                )
            )
    finally:
        mlx_client._CLIENTS.clear()
        mlx_client._CLIENTS.update(original)
    assert client.denials == []
