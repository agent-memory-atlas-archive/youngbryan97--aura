"""A reply that comes before work on a page is not asked for what the page gives.

LIVE 27 Sep: asked to predict her type, take a test on a named site and then
say whether the result matched, her prediction was refused four times as
missing_requested_objective_facets. The facets it lacked were "each answer"
and "whether it matches", which only the test gives, after the reply.
"""

from __future__ import annotations

from types import SimpleNamespace

from interface.routes.chat_turn_recall import _reply_assessment_requires_repair_with_memory_evidence

REQUEST = (
    "Take the Open Extended Jungian Type Scales personality test on openpsychometrics.org. "
    "Before you start, tell me what type you think it will give you and why. As you answer, "
    "say why you chose each answer. When you get your result, tell me whether it matches "
    "what you predicted and whether you think it is accurate."
)
PREDICTION = "Before the browser opens, my prediction: INTP, because my record shows me checking what is true."


def _assessment(*reasons):
    return SimpleNamespace(reasons=tuple(reasons), ok=False, hard_fail=False, repairable=True)


def test_page_work_does_not_need_the_pages_facets_from_the_reply():
    needs = _reply_assessment_requires_repair_with_memory_evidence(
        _assessment("missing_requested_objective_facets"), REQUEST, PREDICTION
    )
    assert needs is False


def test_another_fault_on_page_work_still_needs_repair():
    needs = _reply_assessment_requires_repair_with_memory_evidence(
        _assessment("missing_requested_objective_facets", "internal_task_prompt_leak"), REQUEST, PREDICTION
    )
    assert needs is True


def test_a_question_that_is_not_page_work_still_needs_every_facet():
    asked = "Explain why the sky is blue, verify it with a source, and list three consequences."
    needs = _reply_assessment_requires_repair_with_memory_evidence(
        _assessment("missing_requested_objective_facets"), asked, "Because of scattering."
    )
    assert needs is True
