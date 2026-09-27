"""A person who gets no answer is a thing the health system hears about.

LIVE, 2026-09-21: the same question refused three times with
`canonical_chat_no_reply` while the runtime reported itself HEALTHY
throughout. `_mark_conversation_lane_state` builds the dict that goes back
in the response — it writes `state`, `conversation_ready` and
`last_failure_reason` into a fresh copy and returns it. Nothing durable
sees it, so nothing counted these turns and nothing escalated on a run of
them.

The receipt was already emitted; what was missing was the degradation, and
with it the error budget, which measures distinct degradation classes per
hour and is the thing that notices a bad hour.

Each refusal runs here with an empty reply, and the test reads the
degradation tracker the error budget reads.
"""

from __future__ import annotations

import asyncio

import pytest

from core.runtime.errors import get_degradation_tracker
from interface.routes import chat_lane_state, chat_refusals
from tests.chat_lane_support import patch_chat_lane

QUESTION = "what is the largest file in my downloads folder"


async def _nothing(*_args, **_kwargs):
    return None


@pytest.fixture
def quiet_lane(monkeypatch):
    """Everything a refusal touches besides the record, made inert."""
    get_degradation_tracker().reset()
    patch_chat_lane(
        monkeypatch,
        "_collect_conversation_lane_status",
        lambda *a, **k: {"state": "ready", "conversation_ready": True},
    )
    patch_chat_lane(monkeypatch, "_emit_chat_output_receipt", _nothing)
    patch_chat_lane(monkeypatch, "_log_exchange", _nothing)
    patch_chat_lane(monkeypatch, "_complete_logged_exchange", _nothing)
    patch_chat_lane(monkeypatch, "_self_health_answer_or_empty", lambda *_a, **_k: "")
    yield
    get_degradation_tracker().reset()


def _refuse_canonical() -> object:
    response, _lane, _pending = asyncio.run(
        chat_refusals._refuse_an_empty_canonical_reply(
            _chat_session_id="s",
            _original_user_message=QUESTION,
            _semantic_user_message=QUESTION,
            is_benchmark=False,
            lane={},
            pending_exchange_id=None,
            reply_text="",
        )
    )
    return response


def _records():
    return get_degradation_tracker().recent(subsystem="chat.canonical_reply")


def test_the_empty_canonical_reply_records_a_degradation(quiet_lane):
    response = _refuse_canonical()
    assert getattr(response, "status_code", None) == 200
    assert len(_records()) == 1, (
        "a turn that ends with no answer must reach the degradation sink"
    )


def test_the_empty_benchmark_reply_records_one_too(quiet_lane):
    response, _pending = asyncio.run(
        chat_refusals._refuse_an_empty_benchmark_reply(
            _chat_session_id="s",
            _original_user_message=QUESTION,
            _semantic_user_message=QUESTION,
            final_benchmark_text="",
            pending_exchange_id=None,
        )
    )
    assert getattr(response, "status_code", None) == 502
    assert len(_records()) == 1


def test_the_record_names_what_was_asked(quiet_lane):
    """A count of refusals with no question in it cannot be acted on."""
    _refuse_canonical()
    (record,) = _records()
    assert QUESTION in record.error_message, (
        "the degradation must carry the question, not just the fact"
    )
    assert record.severity == "warning"
    assert record.action


def test_a_turn_that_answered_records_nothing(quiet_lane):
    response, _lane, _pending = asyncio.run(
        chat_refusals._refuse_an_empty_canonical_reply(
            _chat_session_id="s",
            _original_user_message=QUESTION,
            _semantic_user_message=QUESTION,
            is_benchmark=False,
            lane={},
            pending_exchange_id=None,
            reply_text="The largest is a 2.1 GB disk image.",
        )
    )
    from interface.routes.chat import _SEAM_FELL_THROUGH

    assert response is _SEAM_FELL_THROUGH
    assert _records() == []


def test_marking_the_lane_is_not_itself_a_record(quiet_lane):
    """The thing that looked like a record and was not.

    If this ever starts writing somewhere durable, the two can be merged;
    until then the refusal sites own the recording.
    """
    lane = chat_lane_state._mark_conversation_lane_state(
        "canonical_chat_no_reply", state="failed"
    )
    assert lane["state"] == "failed"
    assert lane["last_failure_reason"] == "canonical_chat_no_reply"
    assert get_degradation_tracker().status()["total_degradations"] == 0
