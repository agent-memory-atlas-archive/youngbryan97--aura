"""A page that refused an automated reader says so, wherever it was asked from.

LIVE 2026-09-28, twice in one boot: her curiosity engine asked r/futurology, the
browser's navigation record said `bot_block_or_captcha`, and the adapter
reported "Failed to load r/futurology". A reason that says more tries will not
help is the difference between a retry and a decision, and the reading that
turns one into the other lived in `sovereign_browser` where no other caller
could reach it.
"""
from __future__ import annotations

import pytest

from core.capabilities.phantom_browser import why_it_would_not_load
from core.skills.reddit_adapter import RedditAdapterSkill

pytestmark = pytest.mark.unit


class _Browser:
    def __init__(self, reason: str) -> None:
        self._last_navigation = {"ok": False, "reason": reason} if reason else None


def test_a_bot_block_is_read_as_one():
    said = why_it_would_not_load(_Browser("bot_block_or_captcha"))
    assert "blocking automated browsers" in said
    assert "more tries" in said, "the reason must say retrying will not help"


def test_another_reason_is_passed_through_as_the_browser_put_it():
    assert why_it_would_not_load(_Browser("redirected_elsewhere")) == "redirected_elsewhere"


def test_a_browser_with_no_record_says_nothing_rather_than_guessing():
    assert why_it_would_not_load(_Browser("")) == ""
    assert why_it_would_not_load(object()) == ""


def test_the_adapter_carries_the_reason_into_what_it_reports():
    said = RedditAdapterSkill._could_not_load("r/futurology", _Browser("bot_block_or_captcha"))
    assert said.startswith("Failed to load r/futurology")
    assert "blocking automated browsers" in said


def test_no_reason_leaves_the_plain_sentence():
    said = RedditAdapterSkill._could_not_load("the inbox", _Browser(""))
    assert said == "Failed to load the inbox"


def test_no_load_failure_in_the_adapter_is_left_causeless():
    """Seven sites reported a bare failure; a new one must not join them."""
    import inspect

    source = inspect.getsource(RedditAdapterSkill)
    body = source.split("def _could_not_load", 1)
    rest = body[1].split("\n\n", 4)[-1] if len(body) > 1 else source
    assert 'f"Failed to load' not in rest, (
        "a load failure is reported without the reason the browser already has"
    )
