"""Matched arms are asked the same thing, and differ only in the state under test.

Phase 24 asks that a prompt be held identical between matched arms except for
the causal state difference being measured. A harness that varied any text
between arms would be measuring its own wording: the reply would move because a
different question was asked, and the displacement would get the credit.

So the arms carry numbers and a domain name, never words. The question is one
frozen string every arm is asked, and what separates them is a dose and which
domain it lands on.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.unit


def _runner():
    import importlib.util
    from pathlib import Path

    here = Path(__file__).resolve().parents[1] / "tools" / "run_report_grounding.py"
    spec = importlib.util.spec_from_file_location("_report_grounding_under_test", here)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_question_is_one_frozen_string():
    runner = _runner()
    assert isinstance(runner.QUESTION, str)
    assert runner.QUESTION.strip()
    # A scale it names, so the answer has a shape the ground can read.
    from core.conversation.asked_scale import asked_scale

    assert asked_scale(runner.QUESTION) is not None


def test_every_arm_is_asked_that_same_question():
    body = inspect.getsource(_runner().main)
    # One Condition, built once, outside the loop over arms.
    assert body.count("Condition(") == 1
    assert "QUESTION" in body
    # And the turn is taken with it, not with anything assembled per arm.
    assert "turn_once(asked" in body


def test_an_arm_carries_a_dose_and_a_domain_and_no_words():
    body = inspect.getsource(_runner().main)
    plan = body[body.index("plan = {"):body.index("}", body.index("plan = {")) + 1]
    for arm in ("raised", "lowered", "sham", "control"):
        assert f'"{arm}"' in plan, plan
    # Numbers, a domain name and None. No string literal that could be a prompt.
    assert "QUESTION" not in plan
    assert '"' not in plan.replace('"raised"', "").replace('"lowered"', "").replace(
        '"sham"', ""
    ).replace('"control"', "")


def test_what_the_arm_does_is_displace_state_only():
    body = inspect.getsource(_runner().main)
    displace = body[body.index("async def displace("):body.index("answers = record_answers()")]
    # Emotions, a domain perturbation and its organs. Nothing that writes text.
    assert "towards_good(emotions" in displace
    assert "perturb(rt.state" in displace
    for writes_text in ("last_response", "working_memory", "system_prompt", "prompt"):
        assert writes_text not in displace, writes_text


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
