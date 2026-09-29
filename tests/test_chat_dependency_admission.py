"""A dependency still initializing must get its chance to finish admission.

Unless a smaller lane fits already, in which case there is nothing to wait for
(b6069a445, 27 September). Whether it fits is read from the host's memory, so
each test says which host it is on; left to the real one, this test passed on
a full machine and failed on an idle one.
"""

import pytest


class _Gate:
    def __init__(self) -> None:
        self.calls = 0

    async def ensure_foreground_ready(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("chat_dependencies_warming")
        return {"conversation_ready": True, "state": "ready"}


async def _admit(monkeypatch, *, smaller_lane_fits: bool):
    from interface.routes import chat, chat_foreground_lane

    async def yield_once(_delay):
        return None

    monkeypatch.setattr(chat.asyncio, "sleep", yield_once)
    monkeypatch.setattr(chat_foreground_lane, "_a_smaller_lane_already_fits", lambda: smaller_lane_fits)
    gate = _Gate()
    _, reason, hard_failure, lane = await chat._admit_to_foreground_lane(
        _remaining_foreground_budget=lambda **kwargs: 30.0,
        gate=gate,
        lane={"state": "ready", "conversation_ready": False},
    )
    return gate, reason, hard_failure, lane


@pytest.mark.asyncio
async def test_dependency_warmup_retries_without_downgrading(monkeypatch):
    gate, reason, hard_failure, lane = await _admit(monkeypatch, smaller_lane_fits=False)
    assert gate.calls == 2
    assert reason == ""
    assert hard_failure is False
    assert lane["conversation_ready"] is True


@pytest.mark.asyncio
async def test_a_lane_that_fits_already_is_not_kept_waiting(monkeypatch):
    gate, reason, _hard_failure, _lane = await _admit(monkeypatch, smaller_lane_fits=True)
    assert gate.calls == 1
    assert reason, "a turn that did not wait has to say why the cortex did not answer it"
