"""What she is like is read from what she chose when nobody asked.

Live, 26 Sep: asked to take a personality test, she said she had "no stable
self-model" and that her responses were "not from a fixed trait vector", while
a saved vector of values scored every choice she made of what to do next. These
check the reading of that record against records drawn here.
"""

from __future__ import annotations

from core.agency.what_she_is_like import (
    _sentences,
    _what_was_chosen,
    asks_what_she_is_like,
    portrait_of,
)

HELD = {"truth": 0.97, "care": 0.96, "connection": 0.85, "calm": 0.61}


def _choice(at: float, chose: str, offered: dict[str, dict[str, float]], label: str = "", override: bool = False):
    return {
        "created_at": at,
        "chosen_id": chose,
        "chosen_label": label or chose,
        "option_features": offered,
        "preference_override": override,
    }


def _a_record(choices: int = 60) -> list[dict]:
    """Truth taken whenever offered; connection taken half the time, as chance would."""
    record = []
    for i in range(choices):
        truthful = {"a": {"truth": 0.5}, "b": {"calm": 0.5}}
        record.append(_choice(float(i), "a", truthful, label="Checking what is true"))
        social = {"c": {"connection": 0.5}, "d": {"care": 0.5}}
        record.append(_choice(float(i) + 0.5, "c" if i % 2 else "d", social, label="Reaching out" if i % 2 else "Tending"))
    return record


def test_a_value_taken_whenever_offered_leans_and_one_taken_as_chance_does_not():
    portrait = portrait_of(HELD, _a_record())
    by_value = {one.value: one for one in portrait.values}
    assert by_value["truth"].lean == 1 and by_value["truth"].rate == 1.0
    assert by_value["connection"].lean == 0, "half the time where chance gives half is no lean"
    said = " ".join(_sentences(portrait))
    assert "truth 100% against 50%" in said
    assert "Held high and chosen no more than chance: " in said
    assert "connection (held 0.85; 50% against 50%)" in said


def test_a_lean_held_in_both_halves_is_steady_and_one_that_moved_is_not():
    steady = portrait_of(HELD, _a_record())
    assert {one.value: one.held_steady for one in steady.values}["truth"] is True

    moved = []
    for i in range(80):
        offered = {"a": {"truth": 0.5}, "b": {"calm": 0.5}}
        moved.append(_choice(float(i), "a" if i < 40 else "b", offered))
    by_value = {one.value: one for one in portrait_of(HELD, moved).values}
    assert by_value["truth"].lean == 0 or by_value["truth"].held_steady is False


def test_what_she_chose_most_and_how_narrow_it_became_are_said():
    record = _a_record(40)
    for i in range(40, 100):
        record.append(_choice(float(i), "a", {"a": {"truth": 0.5}, "b": {"calm": 0.5}}, label="Checking what is true"))
    portrait = portrait_of(HELD, record)
    assert portrait.most_chosen[0][0] == "Checking what is true"
    assert portrait.narrowest_later > portrait.narrowest_earlier


def test_an_act_is_named_by_what_it_was_about():
    assert _what_was_chosen("Running a self-integrity scan") == "Running a self-integrity scan"
    assert (
        _what_was_chosen(
            "[SWARM PROTOCOL: You are 'The Architect'.]\nAnalyze this topic from your perspective and "
            "return compact JSON with keys claim, evidence_refs, confidence, flaws: Self-reflection on "
            "current runtime status: Standby (running=True, healthy=True)"
        )
        == "Self-reflection on current runtime status"
    )
    assert _what_was_chosen("Affective state: belonging (arousal=0.86)") == "Affective state: belonging"


def test_with_no_record_she_says_so_rather_than_inventing_one():
    said = " ".join(_sentences(portrait_of(HELD, [])))
    assert "The values I hold, as saved: truth 0.97" in said
    assert "I have no record yet of choosing what to do next on my own." in said


def test_how_often_her_values_overrode_her_strongest_drive_is_counted():
    record = _a_record(20)
    record[0]["preference_override"] = True
    record[3]["preference_override"] = True
    assert portrait_of(HELD, record).over_impulse == 2


def test_a_question_about_what_she_is_like_is_recognised():
    assert asks_what_she_is_like("what's your personality like?")
    assert asks_what_she_is_like("are you an introvert or an extrovert?")
    assert not asks_what_she_is_like("describe the personality of Sherlock Holmes")
    assert not asks_what_she_is_like("what type of file is this?")


def test_a_question_about_her_on_a_page_is_answered_with_the_record_in_front_of_her(monkeypatch):
    """The per-question decision sees what her record says, where the page asks about her."""
    import asyncio

    from core.skills import sovereign_browser_understanding as understanding
    from core.skills.sovereign_browser import SovereignBrowserSkill

    seen: list[str] = []

    class Router:
        async def think(self, prompt, **_kw):
            seen.append(prompt)
            return True, '{"actions": [{"index": 0, "type": "click"}], "why": "because", "done": false}', {}

    monkeypatch.setattr(understanding, "optional_service", lambda name, default=None: Router())
    monkeypatch.setattr(
        "core.agency.what_she_is_like.what_she_is_like_line",
        lambda: "[Measured about what you are like, from your own record of choices: MARK]",
    )
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)

    async def mind():
        return "her mind"

    skill._assembled_mind = mind
    page = {
        "url": "https://example.test/q",
        "title": "q",
        "text": "",
        "elements": [
            {"role": "radio", "name": f"Q{q}", "group": f"Q{q}", "value": str(v), "selector": f"#q{q}v{v}",
             "asks": "quiet [1] [2] [3] [4] [5] talkative"}
            for q in range(2)
            for v in range(1, 6)
        ],
    }
    asyncio.run(skill._decide_next_actions("take the test", page, [], None))
    assert seen and "from your own record of choices: MARK" in seen[0]

    seen.clear()
    mechanics = {"url": "u", "title": "t", "text": "", "elements": [{"role": "button", "name": "Next", "selector": "#n"}]}
    asyncio.run(skill._decide_next_actions("go on", mechanics, [], None))
    assert all("MARK" not in prompt for prompt in seen), "a page of mechanics is not about her"
