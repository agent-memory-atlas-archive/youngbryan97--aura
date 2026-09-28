"""The primary lane's warmup may run while a foreground turn waits for it.

LIVE, 2026-09-21. The boot warmup asked the cortex for a one-token
precompile and then for the visible readiness probe. Both were refused by
``_generate_inner``'s foreground-ownership guard, which returns ``None``
without saying why, so the prover reported ``no_text`` — a claim about the
worker, which had never been asked anything. The lane went to ``recovering``
with ``warmup_readiness_no_text`` and the chat request that was waiting for
readiness got nothing back.

The exemption for the primary lane already existed, one level up, at the
warmup gate in ``mlx_warmup_and_adapters``, with the 2026-07-10 deadlock it
was written for in its comment. It was written there and not at the line that
does the refusing.
"""

from __future__ import annotations

import asyncio
import types

import pytest

from core.brain.llm import mlx_client as mlx
from core.brain.llm.mlx_client import was_declined_before_the_worker


class _Lane:
    """Only the parts of the client the guard reads."""

    model_path = "/models/Aura-Qwen3.8-27B-persona-crsm"

    def __init__(self, *, primary: bool) -> None:
        self._primary = primary
        self._deliberate_no_text_reason: str | None = None
        self.reached_the_worker = False

    def _is_primary_lane(self) -> bool:
        return self._primary

    def _coerce_generation_kwargs(self, *a, **k):
        raise _PastTheGuard

    def __getattr__(self, name: str):
        # Anything the method reaches for after the guard means the guard let
        # it through. Reading a private attribute is not the same as calling
        # the worker, so only a call counts.
        if name.startswith("__"):
            raise AttributeError(name)

        def _reached(*_a, **_k):
            raise _PastTheGuard

        return _reached

    consume_deliberate_no_text_reason = (
        mlx.MLXLocalClient.consume_deliberate_no_text_reason
    )


class _PastTheGuard(Exception):
    """Raised by the first thing ``_generate_inner`` does after the guard."""


async def _past_the_guard(lane: _Lane, **kwargs) -> str | None:
    """Drive the REAL ``_generate_inner`` as far as the guard.

    A double that re-implements the guard would keep passing after somebody
    deleted the exemption, so this calls the shipped method and lets the next
    thing it touches say that control got through.
    """
    try:
        return await mlx.MLXLocalClient._generate_inner(lane, "hi", **kwargs)
    except _PastTheGuard:
        lane.reached_the_worker = True
        return "ready"


@pytest.fixture
def owned_foreground(monkeypatch):
    monkeypatch.setattr(mlx, "_foreground_owner_active", lambda: True)


def test_the_primary_lanes_readiness_probe_is_not_refused(owned_foreground):
    lane = _Lane(primary=True)
    said = asyncio.run(
        _past_the_guard(lane, request_is_background=True, health_probe=True)
    )
    assert said == "ready"
    assert lane.reached_the_worker


def test_the_primary_lanes_precompile_is_not_refused(owned_foreground):
    lane = _Lane(primary=True)
    said = asyncio.run(
        _past_the_guard(lane, request_is_background=True, warmup_precompile=True)
    )
    assert said == "ready"


def test_ordinary_background_work_still_yields_to_the_turn(owned_foreground):
    lane = _Lane(primary=True)
    said = asyncio.run(_past_the_guard(lane, request_is_background=True))
    assert said is None
    assert not lane.reached_the_worker


def test_a_background_lanes_warmup_still_yields_to_the_turn(owned_foreground):
    lane = _Lane(primary=False)
    said = asyncio.run(
        _past_the_guard(lane, request_is_background=True, health_probe=True)
    )
    assert said is None


def test_the_refusal_says_it_was_a_refusal(owned_foreground):
    lane = _Lane(primary=True)
    asyncio.run(_past_the_guard(lane, request_is_background=True))
    assert lane._deliberate_no_text_reason == "skipped_during_foreground_ownership"
    assert was_declined_before_the_worker(lane._deliberate_no_text_reason)


def test_a_worker_that_answered_nothing_is_not_called_a_refusal():
    assert not was_declined_before_the_worker("")
    assert not was_declined_before_the_worker("generation_deadline_worker_healthy")
    assert was_declined_before_the_worker("stopped_before_worker_spawn:cortex_startup_quiet")


def test_the_prover_reports_the_refusal_rather_than_no_text():
    """``prove_visible_readiness`` must not report a refusal as empty output."""
    client = types.SimpleNamespace(
        _deliberate_no_text_reason="skipped_during_foreground_ownership",
        _last_visible_readiness_at=0.0,
    )
    client.is_alive = lambda: True
    client.consume_deliberate_no_text_reason = types.MethodType(
        mlx.MLXLocalClient.consume_deliberate_no_text_reason, client
    )

    async def _declined(*_a, **_k):
        return None

    client._generate_inner = _declined
    client._set_lane_state = lambda *a, **k: None

    proved = asyncio.run(
        mlx.MLXLocalClient.prove_visible_readiness(client, budget_s=1.0)
    )
    assert proved == "declined:skipped_during_foreground_ownership"
    assert client._last_visible_readiness_at == 0.0


class _Warming:
    """A client in the middle of its warmup, recording what it was made to do."""

    model_path = "/models/Aura-Qwen3.8-27B-persona-crsm"

    def __init__(self, *, precompiled: str | None, proved: str) -> None:
        self._precompiled = precompiled
        self._proved = proved
        self.lane_states: list[tuple[str, str]] = []
        self.generations = 0
        self.recoveries = 0
        self._last_ready_at = 0.0
        self._warmup_in_flight = True

    async def _generate_inner(self, *_a, **_k):
        self.generations += 1
        return self._precompiled

    def consume_deliberate_no_text_reason(self) -> str:
        return "skipped_during_foreground_ownership"

    def is_alive(self) -> bool:
        return True

    async def prove_visible_readiness(self, **_k) -> str:
        return self._proved

    def _set_lane_state(self, state: str, reason: str = "") -> None:
        self.lane_states.append((state, reason))

    async def _recover_worker_for_warmup_retry(self) -> None:
        self.recoveries += 1


def _warm(client: _Warming) -> None:
    from core.brain.llm import mlx_warmup_and_adapters as warmup

    with pytest.raises(mlx._WarmupDeferredError):
        asyncio.run(
            warmup._WarmsUpAndSwapsAdapters._run_warmup_precompile(
                client,
                request_is_background=False,
                foreground_request=True,
                owner_name="warmup:test",
                warmup_timeout=120.0,
            )
        )


def test_a_declined_probe_does_not_mark_the_lane_recovering():
    """The warmup stands down; it does not record a failure against her.

    The precompile answered and the readiness probe was declined before the
    worker was asked anything.
    """
    client = _Warming(precompiled="H", proved="declined:skipped_during_foreground_ownership")
    _warm(client)
    assert not [state for state in client.lane_states if state[0] == "recovering"], (
        f"the declined branch marked the lane: {client.lane_states}"
    )
    assert client.recoveries == 0


def test_a_deferral_is_not_retried_as_a_failure():
    """LIVE 2026-09-21: the boot log said "Warmup pre-compile failed once".

    It had not failed. ``_generate_inner`` declined to spawn a worker while
    foreground headroom was reserved, which is the runtime doing what it
    meant to. ``_WarmupDeferredError`` subclasses ``RuntimeError``, so the
    retry loop caught it as an attempt that went wrong, rebooted the worker
    and spent the campaign recovering from a decision.
    """
    client = _Warming(precompiled=None, proved="proved")
    _warm(client)
    assert client.generations == 1, "a deferral was asked again"
    assert client.recoveries == 0, "the worker was rebooted to recover from a decision"


class _Deferred:
    """A client whose precompile the runtime declines, recording what it does."""

    model_path = "/models/Aura-Qwen3.8-27B-persona-crsm"
    _lane_state = "cold"

    def __init__(self) -> None:
        self.lane_states: list[tuple[str, str]] = []
        self.degraded_events: list[str] = []

    def _set_lane_state(self, state: str, reason: str = "") -> None:
        self._lane_state = state
        self.lane_states.append((state, reason))

    def _is_primary_or_deep_lane(self) -> bool:
        return True

    def _is_primary_lane(self) -> bool:
        return True

    def _warmup_timeout(self) -> float:
        return 30.0

    async def _ensure_worker_alive(self, **_k) -> bool:
        return True

    async def _run_warmup_precompile(self, **_k) -> None:
        raise mlx._WarmupDeferredError(
            "stopped_before_worker_spawn:foreground_headroom_reserved"
        )

    def _record_degraded_event(self, name: str, **_k) -> None:
        self.degraded_events.append(name)


@pytest.mark.parametrize("foreground_request", [True, False])
def test_every_precompile_call_stands_down_on_a_deferral(monkeypatch, foreground_request):
    """LIVE 2026-09-28 00:33, one boot: `FAULT RUNTIME-MLX_CLIENT [MARGINAL]
    ... stopped_before_worker_spawn:foreground_headroom_reserved`.

    ``_warmup_impl`` calls the precompile from two branches. The foreground one
    caught ``_WarmupDeferredError`` and stood down; the background one — the
    lane that takes this refusal on nearly every boot — fell through to the
    generic handler, marked the lane recovering and recorded a warning
    degradation and a fault against her for a decision the runtime made on
    purpose. Both branches are run here with the precompile declined.
    """
    import contextlib

    from core.brain.llm import mlx_warmup_and_adapters as warmup

    recorded: list[object] = []

    @contextlib.asynccontextmanager
    async def _owned(*_a, **_k):
        yield None

    monkeypatch.setattr(mlx, "_foreground_owner_context", _owned)
    monkeypatch.setattr(mlx, "_foreground_owner_active", lambda: False)
    monkeypatch.setattr(mlx, "_shutdown_blocks_model_work", lambda *_a, **_k: False)
    monkeypatch.setattr(mlx, "_record_mlx_degradation", lambda exc, **_k: recorded.append(exc))

    client = _Deferred()
    warmed = asyncio.run(
        warmup._WarmsUpAndSwapsAdapters._warmup_impl(
            client, foreground_request=foreground_request
        )
    )
    assert warmed is False
    assert not [s for s in client.lane_states if s[0] == "recovering"], client.lane_states
    assert recorded == [], "a refusal was recorded as a degradation"
    assert client.degraded_events == [], "a refusal was recorded as a fault"
