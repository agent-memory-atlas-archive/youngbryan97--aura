"""A question about her, on a page, is answered by her own model or not at all.

Two defects in one call. The router returns its text, as every other caller
reads it, and the browser unpacked three values from it: every decision that
went to her own reasoning raised, was recorded as a degradation, and came back
failed. And nothing checked which model answered: live on 26 Sep a turn about
a personality test was answered by a 1.5B stand-in while her own model could
not load, and on a page that answer would have been submitted as hers.
"""

from __future__ import annotations

import asyncio

from core.brain.generation_provenance import attributed_text
from core.skills import sovereign_browser_understanding as understanding
from core.skills.sovereign_browser import SovereignBrowserSkill

DECISION = '{"actions": [{"index": 0, "type": "click"}], "why": "I plan ahead", "done": false}'

A_SCALE = {
    "url": "https://example.test/q",
    "title": "q",
    "text": "",
    "elements": [
        {"role": "radio", "name": f"Q{q}", "group": f"Q{q}", "value": str(v), "selector": f"#q{q}v{v}",
         "asks": "plans ahead [1] [2] [3] [4] [5] improvises"}
        for q in range(2)
        for v in range(1, 6)
    ],
}


def _deciding_with(monkeypatch, reply):
    asked: list[dict] = []

    class Router:
        async def think(self, prompt, **kwargs):
            asked.append(kwargs)
            return reply

    monkeypatch.setattr(understanding, "optional_service", lambda name, default=None: Router())
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)

    async def mind():
        return "her mind"

    skill._assembled_mind = mind
    decision = asyncio.run(skill._decide_next_actions("take the test", A_SCALE, [], None))
    return decision, asked


def test_the_routers_text_is_read_as_text(monkeypatch):
    decision, _asked = _deciding_with(monkeypatch, DECISION)
    assert not decision.get("error"), decision
    assert decision["actions"] == [{"index": 0, "type": "click"}]


def test_a_question_about_her_is_asked_of_her_own_model(monkeypatch):
    _decision, asked = _deciding_with(monkeypatch, attributed_text(DECISION, {"endpoint": "Cortex"}))
    assert asked[0].get("prefer_tier") == "primary"
    assert asked[0].get("_non_chat_inference") is True, "a failed generation must come back empty, not as a canned line"


def test_an_answer_from_her_own_model_is_taken(monkeypatch):
    decision, _asked = _deciding_with(monkeypatch, attributed_text(DECISION, {"endpoint": "Cortex"}))
    assert decision["actions"] == [{"index": 0, "type": "click"}]


def test_an_answer_from_a_stand_in_is_not_taken_as_hers(monkeypatch):
    decision, _asked = _deciding_with(monkeypatch, attributed_text(DECISION, {"endpoint": "Reflex"}))
    assert decision.get("error") == "not_her_own_reasoning:Reflex"
    assert not decision.get("actions")


def test_the_old_three_part_reply_is_still_read():
    assert SovereignBrowserSkill._the_text_of((True, DECISION, {})) == DECISION
    assert SovereignBrowserSkill._the_text_of(None) == ""


def test_a_decision_is_held_to_its_shape_by_the_decoder(monkeypatch):
    """Live, 26 Sep: asked with her whole mind in front of her, her own model
    talked about the task ("The user wants me to take...") and chose nothing."""
    _decision, asked = _deciding_with(monkeypatch, attributed_text(DECISION, {"endpoint": "Cortex"}))
    schema = asked[0].get("schema")
    assert schema is SovereignBrowserSkill._DECISION_SCHEMA
    assert set(schema["required"]) >= {"actions", "why"}
    # The schema reached no decoder; the shape is what the worker holds
    # (LIVE 27 Sep 03:32: prose again, no shape held).
    assert asked[0].get("output_shape") == "json_object"


def test_what_she_makes_of_a_page_is_held_to_its_shape(monkeypatch):
    asked: list[dict] = []

    class Router:
        async def think(self, prompt, **kwargs):
            asked.append(kwargs)
            return '{"here": "a test", "to_progress": "answer", "done_when": "a result shows"}'

    monkeypatch.setattr(understanding, "optional_service", lambda name, default=None: Router())
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    made = asyncio.run(skill._understand_page("take the test", A_SCALE, None, "her mind"))
    assert asked and asked[0].get("schema") is SovereignBrowserSkill._UNDERSTANDING_SCHEMA
    assert asked[0].get("output_shape") == "json_object"
    assert made.get("here") == "a test"
