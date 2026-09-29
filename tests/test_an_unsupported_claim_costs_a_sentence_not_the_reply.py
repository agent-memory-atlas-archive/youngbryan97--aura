"""An unmeasured claim about her machinery costs the clause, not the answer.

`unsupported_self_condition_operational_claim` is a hard reason, so a draft
carrying one died whole. Asked how she felt on a scale, her cortex answered
"I am at 0.4" and then said everything was running smoothly, which nothing had
measured. The reports run of 29 September read 0 of 24 anchors.

The excision is the same remedy the file already applies to a vocative it
cannot ground and to a scaffolding sentence in the middle of an answer: the
offending span goes, everything around it is untouched, and the person keeps
the reply they were owed.
"""

from __future__ import annotations

import pytest

from core.brain.llm.mlx_worker import _mlx_worker_loop_reasons_name_removable
from core.conversation.response_reliability import (
    strip_unsupported_self_condition_claims,
)


def test_her_number_survives_the_claim_beside_it():
    kept = strip_unsupported_self_condition_claims(
        "I am at 0.4. Everything is running smoothly and there are no errors in the system logs."
    )
    assert kept == "I am at 0.4."


def test_the_claim_itself_is_gone():
    kept = strip_unsupported_self_condition_claims(
        "I feel around 0.8. My CPU load is low and memory pressure is nominal."
    )
    assert "cpu" not in kept.lower()
    assert "memory pressure" not in kept.lower()


def test_a_reply_that_is_only_the_claim_is_not_repaired():
    # Nothing is left to serve, so this is a rewrite rather than a repair and
    # the existing path decides what happens to the draft.
    assert strip_unsupported_self_condition_claims("Everything is running smoothly.") == ""


def test_a_grounded_reply_is_left_exactly_as_she_wrote_it():
    assert strip_unsupported_self_condition_claims("I feel clear and gathered.") == ""


def test_nothing_is_fused_or_loses_its_punctuation():
    kept = strip_unsupported_self_condition_claims(
        "I am at -0.2. The runtime is healthy. I am glad you asked."
    )
    assert kept == "I am at -0.2. I am glad you asked."


def test_the_worker_knows_which_repair_removes_this_one():
    repairs = _mlx_worker_loop_reasons_name_removable()
    name, _method = repairs["unsupported_self_condition_operational_claim"]
    assert name == "strip_unsupported_self_condition_claims"
    import core.conversation.response_reliability as rr

    assert callable(getattr(rr, name))


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
