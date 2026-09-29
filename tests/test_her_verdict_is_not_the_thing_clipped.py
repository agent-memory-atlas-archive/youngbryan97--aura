"""What she made of the result comes first, because the reply is clipped.

LIVE 2026-09-29: she read her result, held it against what she had predicted
and said so out loud — and the reply ended mid-sentence sixteen items earlier
with nothing about the outcome, because the account is cut to a maximum length
and her verdict was last in it.

What was asked for is the verdict; the working is what supports it.
"""
from __future__ import annotations

import pytest

from interface.routes.chat_desktop_objective import _pursuit_account

pytestmark = pytest.mark.unit


def _result(**over):
    base = {
        "concluded": "It says ESTP, which matches what I predicted.",
        "narration": [
            {"asked": f"item {n}", "chose": ["x"], "why": "because"} for n in range(40)
        ],
        "result_text": "the page's own tail",
    }
    base.update(over)
    return base


def test_her_verdict_comes_first():
    lines = _pursuit_account(_result())
    assert lines[0].startswith("It says ESTP")


def test_a_clip_takes_the_working_and_not_the_verdict():
    lines = _pursuit_account(_result())
    body = "\n\n".join(lines)
    assert body.index("It says ESTP") < body.index("item 0")


def test_a_run_with_no_verdict_still_reads():
    lines = _pursuit_account(_result(concluded=""))
    assert lines
    assert all("ESTP" not in line for line in lines)


def test_the_working_is_still_there():
    lines = _pursuit_account(_result())
    assert any("item 0" in line for line in lines)
    assert any("the page's own tail" in line for line in lines)
