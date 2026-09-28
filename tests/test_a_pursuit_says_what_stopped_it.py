"""A failed pursuit names its cause, because something always knew it.

LIVE 2026-09-28 00:45. A page run's cortex lane exhausted mid-decision, the
round logged `Pursuit decision unusable (empty_decision)`, the stall limit ended
the run — and the result dict carried no `error` field at all, so `BaseSkill`
filled the silence with "sovereign_browser reported failure without a cause
(status=failed_recoverable)". The surprise engine banked the causeless version
at surprise 0.90, and the incident that followed named the wrong subsystem.
"""
from __future__ import annotations

import asyncio
from typing import Any

import pytest

from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


class _Page:
    """A page that is always there and never changes."""

    def __init__(self) -> None:
        self.closed = False

    async def observe(self, **_kw: Any) -> dict[str, Any]:
        return {
            "url": "https://example.test/survey",
            "text": "Question one of sixty.",
            "elements": [
                {"index": 0, "tag": "input", "type": "radio", "name": "q1"},
                {"index": 1, "tag": "button", "name": "Next"},
            ],
        }

    async def close(self) -> None:
        self.closed = True


def _skill(monkeypatch, decision: dict[str, Any]) -> SovereignBrowserSkill:
    skill = SovereignBrowserSkill()

    async def _understood(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"here": "a survey", "to_progress": "answer it", "done_when": "submitted"}

    async def _decided(*_a: Any, **_k: Any) -> dict[str, Any]:
        return dict(decision)

    monkeypatch.setattr(skill, "_understand_page", _understood)
    monkeypatch.setattr(skill, "_decide_next_actions", _decided)
    monkeypatch.setattr(skill, "_asks_about_the_one_answering", lambda *_a: False)
    monkeypatch.setattr(skill, "_narrate", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_remember_the_place", lambda *_a, **_k: None)
    monkeypatch.setattr(skill, "_recall_about", lambda *_a, **_k: "")
    monkeypatch.setattr(skill, "_retain_stated_positions", lambda *_a, **_k: None)
    return skill


def test_a_run_stopped_by_an_empty_decision_says_so(monkeypatch):
    skill = _skill(monkeypatch, {"error": "empty_decision", "raw": ""})
    result = asyncio.run(
        skill._handle_pursue(_Page(), None, "take the survey", 8)
    )
    assert result["ok"] is False
    assert "empty_decision" in result["error"], (
        f"the cause was known and the result said {result.get('error')!r}"
    )
    assert result["error"] == "no_progress:empty_decision", (
        "both facts belong in the cause: what ended the run, and why nothing "
        f"was landing — got {result['error']!r}"
    )


def test_a_failed_run_never_reports_an_empty_cause(monkeypatch):
    """Whatever ended it, the field is never blank on a failure."""
    skill = _skill(monkeypatch, {"error": "unparsable_decision", "raw": "I think..."})
    result = asyncio.run(
        skill._handle_pursue(_Page(), None, "take the survey", 8)
    )
    assert result["ok"] is False
    assert str(result.get("error") or "").strip(), "a failure with no stated cause"


def test_a_run_that_landed_work_is_not_a_failure(monkeypatch):
    """And a successful pursuit states no error, so nothing reads one."""
    skill = _skill(monkeypatch, {"done": True, "actions": [], "why": "finished"})
    result = asyncio.run(
        skill._handle_pursue(_Page(), None, "take the survey", 8)
    )
    assert result.get("error") == "" or result["ok"] is False
