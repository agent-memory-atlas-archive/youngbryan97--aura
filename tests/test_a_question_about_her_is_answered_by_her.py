"""Who answers is part of what a self-report asks.

LIVE 2026-09-28. A personality test asked her thirty-two questions about
herself. Every item requested the primary tier, because only she can answer one
— and morphogenesis advised a tier downgrade at 0.92 resource pressure, so all
thirty-two went to the brainstem, which chose the middle option every time.

Two things let that happen. Routing treated an explicit tier as a preference it
could overrule under load. And the guard meant to refuse a stand-in read
provenance off the reply, which the router returns as a plain string — so it saw
"" on every call and passed. A check that cannot see is a check that always
passes.
"""
from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest

from core.brain.request_contract import POLICY_FIELDS, REQUEST_FIELDS, validate_request_context
from core.skills.sovereign_browser import SovereignBrowserSkill

pytestmark = pytest.mark.unit


def test_the_requirement_is_a_declared_policy_field():
    assert "own_lane_required" in REQUEST_FIELDS
    assert "own_lane_required" in POLICY_FIELDS
    context = validate_request_context({"own_lane_required": True}).context
    assert context.get("own_lane_required") is True


def test_a_malformed_requirement_is_rejected_rather_than_guessed():
    validation = validate_request_context({"own_lane_required": "maybe"})
    assert "own_lane_required" in validation.rejected


def test_load_cannot_downgrade_a_lane_the_caller_requires():
    """The advice is about pressure; it cannot answer as somebody else."""
    from core.brain import inference_gate_turn_setup

    body = inspect.getsource(
        inference_gate_turn_setup._SetsUpTheTurn._generate_with_metadata_sink_part_5
        if hasattr(inference_gate_turn_setup, "_SetsUpTheTurn")
        else inference_gate_turn_setup
    )
    advice = body.index("get_morphogenesis_routing_advice")
    guard = body.index("own_lane_required")
    assert guard < advice, (
        "the requirement must be read before the downgrade decides anything"
    )


def _decider(monkeypatch, reply: Any, sink_fill: dict[str, Any] | None):
    skill = SovereignBrowserSkill()
    seen: dict[str, Any] = {}

    async def _think(prompt: str, **kwargs: Any) -> Any:
        seen.update(kwargs)
        sink = kwargs.get("_generation_metadata_sink")
        if isinstance(sink, dict) and sink_fill is not None:
            sink.update(sink_fill)
        return reply

    class _Router:
        think = staticmethod(_think)

    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda name, default=None: _Router() if name == "llm_router" else default,
    )

    async def _mind() -> str:
        return "her mind"

    monkeypatch.setattr(skill, "_assembled_mind", _mind)
    return skill, seen


def test_the_call_declares_that_it_needs_her_own_lane(monkeypatch):
    skill, seen = _decider(
        monkeypatch, '{"actions": [], "why": "x", "done": false}', {"endpoint": "Cortex"}
    )
    asyncio.run(
        skill._decide_next_actions("take the test", {"elements": []}, [], about_her=True)
    )
    assert seen.get("own_lane_required") is True
    assert isinstance(seen.get("_generation_metadata_sink"), dict)


def test_a_stand_in_answer_is_refused(monkeypatch):
    skill, _seen = _decider(
        monkeypatch, '{"actions": [], "why": "x", "done": false}', {"endpoint": "Brainstem"}
    )
    decision = asyncio.run(
        skill._decide_next_actions("take the test", {"elements": []}, [], about_her=True)
    )
    assert decision.get("error", "").startswith("not_her_own_reasoning:Brainstem")


def test_an_unattributed_answer_is_refused(monkeypatch):
    """Nothing saying who answered is not evidence that she did."""
    skill, _seen = _decider(
        monkeypatch, '{"actions": [], "why": "x", "done": false}', {}
    )
    decision = asyncio.run(
        skill._decide_next_actions("take the test", {"elements": []}, [], about_her=True)
    )
    assert decision.get("error") == "not_her_own_reasoning:unattributed"


def test_her_own_answer_is_taken(monkeypatch):
    skill, _seen = _decider(
        monkeypatch, '{"actions": [], "why": "mine", "done": false}', {"endpoint": "Cortex"}
    )
    decision = asyncio.run(
        skill._decide_next_actions("take the test", {"elements": []}, [], about_her=True)
    )
    assert not decision.get("error")
    assert decision.get("why") == "mine"

def test_a_self_report_never_falls_through_to_a_bare_model_call(monkeypatch):
    """LIVE 2026-09-28: the branch was gated on her assembled mind.

    With no state service the assembler returns nothing, and every self-report
    item fell to `generate(prompt)` at the bottom of the function: no identity,
    no self-knowledge, no record of what she is like, and no lane requirement.
    That is what answered thirty-two questions about her, and it answered the
    midpoint every time because it had nothing to prefer with.
    """
    bare: list[str] = []
    skill, _seen = _decider(
        monkeypatch, '{"actions": [], "why": "mine", "done": false}', {"endpoint": "Cortex"}
    )

    async def _no_mind() -> str:
        return ""

    monkeypatch.setattr(skill, "_assembled_mind", _no_mind)
    decision = asyncio.run(
        skill._decide_next_actions("take the test", {"elements": []}, [], about_her=True)
    )
    assert not bare, "a question about her reached a bare model call"
    assert decision.get("why") == "mine"


def test_with_nothing_that_can_answer_as_her_the_item_is_refused(monkeypatch):
    skill = SovereignBrowserSkill()

    class _Router:
        async def generate(self, prompt: str, **_kw: Any) -> str:
            raise AssertionError("a bare call answered a question about her")

    monkeypatch.setattr(
        "core.skills.sovereign_browser_understanding.optional_service",
        lambda name, default=None: _Router() if name == "llm_router" else default,
    )

    async def _no_mind() -> str:
        return ""

    monkeypatch.setattr(skill, "_assembled_mind", _no_mind)
    decision = asyncio.run(
        skill._decide_next_actions("take the test", {"elements": []}, [], about_her=True)
    )
    assert decision.get("error") == "not_her_own_reasoning:no_lane"


def test_she_has_measured_preferences_to_answer_from():
    """Leaning one way needs something to lean with."""
    from core.agency.what_she_is_like import what_she_is_like_line

    said = what_she_is_like_line()
    assert said, "her record of choices says nothing about what she is like"
    assert "values I hold" in said or "chose most" in said
