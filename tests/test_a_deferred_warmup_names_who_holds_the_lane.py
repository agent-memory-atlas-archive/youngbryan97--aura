"""A warm-up that is refused a worker says who refused it.

LIVE 2026-09-28: a training job took the exclusive model lane at 17:11 and
the live instance booted at 17:20. Model-load admission refused the cortex
with `exclusive_lane_owned:standalone:60678:semantic-native:...`, and the
warm-up then wrote `warmup_deferred` over that reason, so the health pulse
reported "conversation_lane: recovering (warmup_deferred)" for forty minutes
without naming the job that held the lane.
"""
from __future__ import annotations

import asyncio

import pytest

from core.brain.llm import mlx_client
from core.brain.llm.mlx_client import MLXLocalClient

HELD = "exclusive_lane_owned:standalone:60678:semantic-native:joint-graph-v7"


def _refused_client(monkeypatch, cause: str) -> MLXLocalClient:
    client = MLXLocalClient(model_path="/models/Qwen2.5-7B-Instruct-4bit")
    monkeypatch.setattr(mlx_client, "_shutdown_blocks_model_work", lambda *_a, **_k: False)

    async def refused(**_kwargs):
        if cause:
            client._set_lane_state("recovering", cause)
        return False

    monkeypatch.setattr(client, "_ensure_worker_alive", refused)
    return client


@pytest.mark.parametrize("foreground", [True, False])
def test_the_lane_keeps_the_reason_admission_gave(monkeypatch, foreground):
    client = _refused_client(monkeypatch, HELD)
    assert asyncio.run(client._warmup_impl(foreground_request=foreground)) is False
    assert client._lane_state == "recovering"
    assert client._lane_error == f"warmup_deferred:{HELD}"


@pytest.mark.parametrize("foreground", [True, False])
def test_every_reader_still_sees_a_lane_that_is_waiting(monkeypatch, foreground):
    from core.brain.llm_health_router import _only_warming
    from core.runtime.errors import backpressure_markers

    client = _refused_client(monkeypatch, HELD)
    asyncio.run(client._warmup_impl(foreground_request=foreground))
    assert _only_warming(client._lane_error)
    assert any(marker in client._lane_error for marker in backpressure_markers())


@pytest.mark.parametrize("foreground", [True, False])
def test_a_refusal_with_no_reason_is_still_called_a_deferral(monkeypatch, foreground):
    client = _refused_client(monkeypatch, "")
    assert asyncio.run(client._warmup_impl(foreground_request=foreground)) is False
    assert client._lane_state == "recovering"
    assert client._lane_error == "warmup_deferred"


def test_a_reason_from_an_earlier_attempt_is_not_carried_into_this_one(monkeypatch):
    client = _refused_client(monkeypatch, "")
    client._set_lane_state("recovering", "exclusive_lane_owned:an_owner_long_gone")
    assert asyncio.run(client._warmup_impl(foreground_request=False)) is False
    assert client._lane_error == "warmup_deferred"
