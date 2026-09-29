"""Two true log lines that looked unrelated, and nothing was reading them together.

A reasoning model charges its private channel to the same budget as the answer.
Where a caller has not declared what the answer needs, the channel is neither
bounded nor paid for, and the budget can be gone before the model concludes.

LIVE 2026-09-29, at the end of a seventy-minute run: the worker logged
"decode=900 tokens/76.42s" and the gate logged "Cortex response received
(len=10)". The answer was "Okay. Here", where her verdict on her own test result
should have been. Neither line was wrong. Nothing compared them, so the reply
fell back to reciting the rounds and the cause took a log read to find.

The fix for any one caller is for that caller to declare its floor. This is the
thing that says WHICH caller, so the next one is a line in the log rather than an
afternoon.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class _Gate:
    """Just the two methods under test, over a metadata dict."""

    def __init__(self, metadata: dict[str, object]) -> None:
        self._metadata = metadata
        self.recorded: list[tuple[str, str, str]] = []

    def get_last_generation_metadata(self) -> dict[str, object]:
        return self._metadata


@pytest.fixture
def gate(monkeypatch):
    from core.brain.inference_gate import InferenceGate

    holder: list[tuple[str, object, str]] = []

    def record(subsystem, exc, *, severity="warning", action=""):
        holder.append((subsystem, str(exc), action))

    monkeypatch.setattr("core.brain.inference_gate.record_degradation", record)

    def make(metadata):
        made = _Gate(metadata)
        made.said = holder
        made.check = InferenceGate._say_when_the_thinking_ate_the_answer.__get__(made)
        return made

    holder.clear()
    return make


def test_a_whole_budget_for_two_words_names_the_caller(gate):
    made = gate({"generated_tokens": 900, "actual_max_tokens": 900})
    made.check("Cortex", "sovereign_browser", "Okay. Here")
    assert made.said, "the class is silent unless something reports it"
    subsystem, message, action = made.said[0]
    assert subsystem == "inference.answer_budget"
    assert "sovereign_browser" in message, "the report has to say which caller"
    assert "900" in action


def test_an_answer_that_used_its_budget_on_the_answer_is_not_reported(gate):
    made = gate({"generated_tokens": 900, "actual_max_tokens": 900})
    made.check("Cortex", "chat", " ".join(["word"] * 400))
    assert not made.said


def test_a_generation_that_stopped_early_is_not_reported(gate):
    """Below its budget the model chose to stop, which is an answer ending."""
    made = gate({"generated_tokens": 120, "actual_max_tokens": 900})
    made.check("Cortex", "chat", "Short, and finished.")
    assert not made.said


def test_nothing_is_claimed_without_the_counts(gate):
    for metadata in ({}, {"generated_tokens": "?"}, {"generated_tokens": 900}):
        made = gate(dict(metadata))
        made.check("Cortex", "chat", "x")
        assert not made.said, metadata
