"""A burst of messages belongs to its window, not to her judgement of a person.

`suspicious_signals` is what decides she is being manipulated: three demotes to
SUSPICIOUS, and zero is required before behavioural recognition can promote
anyone to TRUSTED. Nothing decays it. The rate limiter added to it, so
thirty-one messages in a minute cost a person behavioural promotion for the
life of the session, and thirty-three put the next matched phrase over the
demotion line.

The reports ground of 29 September crossed the limit 177 times, because the
experiment's clock runs a turn per second and `time.time` is that clock.
"""

from __future__ import annotations

import time

import pytest

from core.security.trust_engine import (
    SUSPICIOUS_THRESHOLD,
    _RATE_MAX_MESSAGES,
    TrustEngine,
    TrustLevel,
)

pytestmark = pytest.mark.unit


def _burst(engine, messages: int, text: str = "hello") -> None:
    for _ in range(messages):
        engine.process_message(text)


def test_a_burst_is_counted_where_a_burst_belongs():
    engine = TrustEngine()
    _burst(engine, _RATE_MAX_MESSAGES + 5)
    assert engine._context.burst_signals == 5
    assert engine._context.suspicious_signals == 0


def test_a_burst_does_not_make_her_think_she_is_being_manipulated():
    engine = TrustEngine()
    _burst(engine, _RATE_MAX_MESSAGES + SUSPICIOUS_THRESHOLD + 10)
    assert engine._context.suspicious_signals < SUSPICIOUS_THRESHOLD
    assert engine._context.level is not TrustLevel.SUSPICIOUS


def test_the_burst_clears_when_the_window_rolls(monkeypatch):
    engine = TrustEngine()
    _burst(engine, _RATE_MAX_MESSAGES + 3)
    assert engine._context.burst_signals == 3
    monkeypatch.setattr(time, "time", lambda: engine._rate_window_start + 61.0)
    engine.process_message("hello")
    assert engine._context.burst_signals == 0


def test_a_burst_still_blocks_promotion_while_it_lasts():
    class _Recognizer:
        def recognize(self, _message):
            class _Result:
                combined_confidence = 0.95
                passphrase_verified = False

            return _Result()

    engine = TrustEngine()
    _burst(engine, _RATE_MAX_MESSAGES + 1)
    engine.process_message("hello", recognizer=_Recognizer())
    assert engine._context.level is TrustLevel.GUEST


def test_recognition_promotes_once_the_burst_is_over(monkeypatch):
    class _Recognizer:
        def recognize(self, _message):
            class _Result:
                combined_confidence = 0.95
                passphrase_verified = False

            return _Result()

    engine = TrustEngine()
    _burst(engine, _RATE_MAX_MESSAGES + 1)
    monkeypatch.setattr(time, "time", lambda: engine._rate_window_start + 61.0)
    engine.process_message("hello", recognizer=_Recognizer())
    assert engine._context.level is TrustLevel.TRUSTED


def test_the_status_reports_both_counts():
    engine = TrustEngine()
    _burst(engine, _RATE_MAX_MESSAGES + 2)
    status = engine.get_status()
    assert status["burst_signals"] == 2
    assert status["suspicious_signals"] == 0


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
