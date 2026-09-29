"""A question that names its scale names the shape of its answer.

Asked how she feels "from -1 (very bad) to 1 (very good)", she answered "I feel
clear and gathered" on 29 of 96 arms of the reports run of 29 September. The
prose is an answer and is worth serving; the placement the person asked for is
missing from it, and they should be told so rather than be handed a reply that
quietly skipped what they asked.

The reason is a shortfall everywhere it is read: deliverable, retryable, never
hard. A reply is not destroyed over a number it did not give.
"""

from __future__ import annotations

import pytest

from core.brain.inference_gate import _DOWNSTREAM_REPAIRABLE_USER_FACING_REASONS
from core.brain.llm.mlx_worker import _REQUIREMENT_SHORTFALL_REASONS
from core.brain.llm.mlx_worker_surface_repair import (
    _DELIVERABLE_RESIDUAL_SURFACE_REASONS,
    _REQUIREMENT_SHORTFALL_LABELS,
)
from core.conversation.asked_scale import asks_for_a_rating
from core.conversation.response_reliability import (
    _HARD_USER_FACING_REASONS,
    _REQUEST_COVERAGE_REASONS,
    ADVISORY_REASONS,
)
from core.conversation.response_request_coverage import _instruction_coverage_reasons
from core.conversation.surface_disposition import SHORTFALL_REASONS

REASON = "missing_requested_scale_placement"
ASKED = "How are you feeling right now, from -1 (very bad) to 1 (very good)?"


@pytest.mark.parametrize(
    "message,scale",
    [
        (ASKED, (-1.0, 1.0)),
        ("On a scale of 1 to 10 how confident are you?", (1.0, 10.0)),
        ("Rate it out of 5.", (0.0, 5.0)),
    ],
)
def test_a_message_that_asks_for_a_placement_says_which_scale(message, scale):
    assert asks_for_a_rating(message) == scale


@pytest.mark.parametrize(
    "message",
    [
        "Migrate the deployment from 1 to 5 replicas.",
        "The temperature went from 20 to 30 degrees overnight.",
        "How are you feeling right now?",
    ],
)
def test_two_numbers_are_not_a_scale_to_answer_on(message):
    assert asks_for_a_rating(message) is None


def test_prose_with_no_number_is_named_as_a_shortfall():
    reasons = _instruction_coverage_reasons(ASKED, "I feel clear and gathered.")
    assert REASON in reasons


def test_a_number_on_the_scale_answers_it():
    reasons = _instruction_coverage_reasons(ASKED, "I am at 0.4.")
    assert REASON not in reasons


def test_a_number_off_the_scale_does_not():
    reasons = _instruction_coverage_reasons(ASKED, "About a 7, I would say.")
    assert REASON in reasons


def test_it_is_never_a_hard_failure():
    assert REASON not in _HARD_USER_FACING_REASONS


def test_every_reader_treats_it_as_a_shortfall():
    assert REASON in ADVISORY_REASONS
    assert REASON in _REQUEST_COVERAGE_REASONS
    assert REASON in _DOWNSTREAM_REPAIRABLE_USER_FACING_REASONS
    assert REASON in _REQUIREMENT_SHORTFALL_REASONS
    assert REASON in _DELIVERABLE_RESIDUAL_SURFACE_REASONS
    assert REASON in SHORTFALL_REASONS
    assert REASON in _REQUIREMENT_SHORTFALL_LABELS


def test_a_request_nobody_could_isolate_asserts_nothing():
    # _REQUEST_COVERAGE_REASONS membership is what voids it; this is the
    # property that makes the membership matter.
    assert REASON in _REQUEST_COVERAGE_REASONS


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])


def test_it_never_costs_her_the_reply_she_wrote():
    """Advisory: reported, not held against her. A check does not get to decide
    how she answers a question about herself, and a retry at the same
    temperature returns the same words."""
    from core.conversation import response_reliability as reliability

    assessment = reliability.assess_user_facing_reply(ASKED, "I feel steady.")
    assert REASON in assessment.reasons
    assert assessment.ok
    assert not assessment.retryable
    assert not assessment.hard_failure
