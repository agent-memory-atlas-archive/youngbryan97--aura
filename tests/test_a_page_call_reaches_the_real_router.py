"""A page decision reaches the model through the router the runtime uses.

LIVE 27 Sep: every page decision of a personality test raised before any model
saw it, "got multiple values for keyword argument '_non_chat_inference'", and a
bare fallback wrote prose where a decision belonged. The browser's own tests
passed throughout, because their router was a stand-in that took any
arguments. These use the router class the runtime registers.
"""

from __future__ import annotations

import asyncio

from core.brain.llm_health_router import HealthAwareLLMRouter
from core.skills import sovereign_browser_understanding as understanding
from core.skills.sovereign_browser import SovereignBrowserSkill

DECISION = '{"actions": [{"index": 0, "type": "click"}], "why": "the way in", "done": false}'


def _router(seen: dict):
    router = HealthAwareLLMRouter.__new__(HealthAwareLLMRouter)

    async def generate_with_metadata(**kwargs):
        seen.update(kwargs)
        return {"ok": True, "text": DECISION, "endpoint": "Cortex"}

    router.generate_with_metadata = generate_with_metadata
    router._publish_generation_metadata = lambda *args, **kwargs: None
    return router


def test_saying_non_chat_to_think_does_not_break_the_call():
    seen: dict = {}
    out = asyncio.run(_router(seen).think("p", system_prompt="m", _non_chat_inference=True, output_shape="json_object"))
    assert out
    assert seen["_non_chat_inference"] is True and seen["output_shape"] == "json_object"


def test_a_page_decision_arrives_with_its_shape_and_origin(monkeypatch):
    seen: dict = {}
    monkeypatch.setattr(understanding, "optional_service", lambda name, default=None: _router(seen))
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)

    async def mind():
        return "her mind"

    skill._assembled_mind = mind
    page = {
        "url": "u", "title": "t", "text": "",
        "elements": [
            {"role": "radio", "name": f"Q{q}", "group": f"Q{q}", "value": str(v), "selector": f"#q{q}v{v}",
             "asks": "quiet [1] [2] [3] talkative"}
            for q in range(2)
            for v in range(1, 4)
        ],
    }
    decision = asyncio.run(skill._decide_next_actions("take the test", page, [], None))
    assert not decision.get("error"), decision
    assert seen.get("output_shape") == "json_object"
    assert seen.get("origin") == "sovereign_browser"
