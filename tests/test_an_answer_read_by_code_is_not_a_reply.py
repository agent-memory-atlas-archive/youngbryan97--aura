"""A generation that picks a move is not a reply to anybody.

Graded as one, it fails against a question invented from its own prompt.
LIVE 2026-08-27, on the first move of a pursuit: five reasons at once —
numeric_answer_missing, unanswered_question_part,
off_topic_self_reflection_reply, missing_requested_self_process_coverage,
missing_requested_objective_facets — none of them about the thing it was
actually asked for, which was a direction to press.

The draft was then thrown away as a failed generation, the local fallback
returned nothing, and she decided without language on a board she could read
perfectly well.

The caller already says which it is. her_reasoning has passed
internal_inference=True since the last time this bit, and every other layer
honours it. This one was working out user visibility from where the request
came from rather than from what the answer is for — and a move decision inside
a foreground task is foreground, and is not a reply.
"""

from __future__ import annotations

import inspect
import re

import pytest


def visibility_block() -> str:
    """The lines that decide whether a draft is judged as a user-facing reply."""
    import core.brain.inference_gate as gate

    source = inspect.getsource(gate)
    found = re.search(r"is_user_visible = bool\((.*?)\n            \)", source, re.DOTALL)
    assert found, "the user-visibility decision has moved or been renamed"
    return found.group(1)


def test_an_internal_inference_is_not_judged_as_a_reply():
    assert "internal_inference" in visibility_block()


@pytest.mark.parametrize(
    "excluded",
    ["health_probe", "proof_evaluation_contract", "strict_output_contract",
     "web_interlocutor_contract"],
)
def test_and_everything_that_was_already_excluded_still_is(excluded):
    assert excluded in visibility_block()


def test_her_reasoning_still_says_which_it_is():
    """The flag is only worth reading if the decision lane still sets it."""
    import core.agency.her_reasoning as her

    assert "internal_inference=True" in inspect.getsource(her)


def test_and_still_keeps_the_thinking_channel_shut():
    """The other half of the same defect, from 2026-08-26."""
    import core.agency.her_reasoning as her

    assert 'cognitive_mode="fast"' in inspect.getsource(her)


class _Answers:
    """A client whose one generation is a plan cut off at its budget."""

    PLAN = (
        "LINE: keep the largest tile in the bottom-left corner and build along "
        "the bottom row. ENDS WHEN: the corner tile has to move out of the "
        "corner, which happens when the bottom row"
    )

    async def generate_text_async(self, prompt, **_kwargs):
        return (True, self.PLAN, {})


def _asked(*, internal_inference: bool, foreground_request: bool):
    import asyncio

    from core.brain.inference_gate import InferenceGate
    from core.utils.deadlines import get_deadline

    return asyncio.run(
        InferenceGate()._generate_with_client(
            _Answers(),
            "how will you play this game",
            "",
            [],
            get_deadline(10.0),
            "Cortex",
            messages=[{"role": "user", "content": "how will you play this game"}],
            origin="agency_settling_on_an_approach",
            foreground_request=foreground_request,
            internal_inference=internal_inference,
        )
    )


@pytest.mark.parametrize("foreground_request", [True, False])
def test_an_answer_for_code_that_ran_into_its_budget_goes_to_its_reader(foreground_request):
    """A plan cut at its budget still names its line, and its reader parses that.

    LIVE 2026-09-23: a plan of 1,138 characters was refused as truncated_tail
    after a hundred seconds of the model's time, the lane was counted as
    failed, and the worker that wrote it was killed as stuck.
    """
    said = _asked(internal_inference=True, foreground_request=foreground_request)
    assert said == _Answers.PLAN


def test_the_same_cut_text_is_still_refused_when_nobody_reads_it_as_code():
    """The null: without the flag, a background draft cut mid-clause is malformed."""
    assert _asked(internal_inference=False, foreground_request=False) is None
