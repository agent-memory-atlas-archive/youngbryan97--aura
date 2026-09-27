"""A reply refused as a prompt leak says which sentence was the leak.

LIVE 27 Sep: three drafts in a row were refused as internal_task_prompt_leak,
each opening with her prediction, and the log held only the first 280
characters of each. Nothing said what the detector had read.
"""

from __future__ import annotations

from core.conversation.response_reliability import internal_task_prompt_leak_evidence


def test_the_sentence_that_leaked_is_the_one_named():
    said = "My prediction is INTP. We need answer user question about the test. Then I start."
    assert internal_task_prompt_leak_evidence(said, "take the test") == (
        "We need answer user question about the test."
    )


def test_a_reply_that_is_not_a_leak_names_nothing():
    assert internal_task_prompt_leak_evidence("My prediction is INTP. I plan ahead.", "take the test") == ""


def test_an_unfounded_tool_claim_names_its_sentence():
    from core.conversation.response_reliability import unfounded_tool_execution_claim_evidence

    said = "I expect INTP. I ran the code and the output is: 5."
    assert unfounded_tool_execution_claim_evidence(said, None) == "I ran the code and the output is: 5."
    assert unfounded_tool_execution_claim_evidence("I expect INTP.", None) == ""
