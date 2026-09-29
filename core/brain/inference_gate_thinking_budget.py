"""When the thinking a turn was allowed spent the whole budget and left no answer, and which caller set that budget.

Lifted whole out of `inference_gate`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations



class _ThinkingBudgetMixin:
    """Lifted whole out of InferenceGate; see inference_gate.py."""

    def _say_when_the_thinking_ate_the_answer(
        self, label: str, origin: str, cleaned: str
    ) -> None:
        """Name the caller when a whole token budget produced a stub of an answer.

        A reasoning model charges its private channel to the same budget as the
        answer. Where a caller has not declared what the answer needs, the channel
        is neither opened under a bound nor paid for, the model searches wherever
        it likes, and the budget can be gone before it concludes. What reaches the
        log is two true lines that look unrelated: the worker reporting that it
        decoded every token it was given, and the gate reporting a short answer.

        LIVE 2026-09-29: "decode=900 tokens/76.42s" and "Cortex response received
        (len=10)" — ten characters, "Okay. Here", where her verdict on her own
        result should have been. Nothing was wrong with either line and nothing
        was watching them together, so the reply fell back to reciting the rounds
        and the cause took a log read to find.

        This is that comparison, once, where both numbers are already known. It
        does not fix the caller — the fix is for the caller to declare its floor —
        but it says which caller, and the class is silent otherwise.
        """
        from .inference_gate import (
            record_degradation,
        )

        metadata = self.get_last_generation_metadata()
        if not metadata:
            return
        try:
            decoded = int(metadata.get("generated_tokens") or 0)
            allowed = int(
                metadata.get("actual_max_tokens")
                or metadata.get("requested_max_tokens")
                or 0
            )
        except (TypeError, ValueError):
            # not a failure: a receipt whose counts will not parse cannot be
            # compared, and this reports nothing rather than guessing.
            return
        if decoded <= 0 or allowed <= 0 or decoded < allowed:
            return
        # Tokens the answer actually carries, against the tokens it was charged.
        # A budget spent in full on an answer that is a fraction of it went
        # somewhere else, and the private channel is the only other place.
        said = len(str(cleaned or "").split())
        if said >= decoded / 4:
            return
        record_degradation(
            "inference.answer_budget",
            RuntimeError(
                f"thinking_spent_the_answer_budget:{origin or 'unattributed'}:"
                f"{decoded}_tokens_for_{said}_words"
            ),
            severity="warning",
            action=(
                f"{label} decoded its whole {allowed}-token budget for a "
                f"{said}-word answer; the caller declares no completion floor, "
                "so the private channel was unbounded"
            ),
        )

