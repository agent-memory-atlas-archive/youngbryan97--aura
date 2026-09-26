"""A question is what the page shows around its options, not what the options are called.

Found on a live personality inventory, 26 Sep: sixty questions on one page, three
hundred radios with no label, no aria text and no title. Each option's name
fell back to its group's own name, so every question looked like one option
of its own, the page was not taken for a scale instrument, and the decision
for "question Q1" had nothing to say what Q1 asked. What it asked was the row:
a phrase, five radios, and the opposite phrase; and for the second half, a
statement under a heading row naming the scale.

These are three common layouts drawn here, read by the real observation
script in a real headless browser.
"""

from __future__ import annotations

import asyncio

import pytest

from core.capabilities.phantom_browser import PhantomBrowser
from core.skills.sovereign_browser import SovereignBrowserSkill

A_SCALE_BETWEEN_TWO_PHRASES = """
<form><table>
  <tr><td>plans ahead</td>
      <td><input type=radio name=A1 value=1></td><td><input type=radio name=A1 value=2></td>
      <td><input type=radio name=A1 value=3></td><td><input type=radio name=A1 value=4></td>
      <td><input type=radio name=A1 value=5></td><td>improvises</td></tr>
  <tr><td>quiet</td>
      <td><input type=radio name=A2 value=1></td><td><input type=radio name=A2 value=2></td>
      <td><input type=radio name=A2 value=3></td><td><input type=radio name=A2 value=4></td>
      <td><input type=radio name=A2 value=5></td><td>talkative</td></tr>
</table><input type=submit value=Next></form>
"""

A_GRID_UNDER_A_HEADING = """
<form><table>
  <tr><td></td><td colspan=5>Never &nbsp; Sometimes &nbsp; Always</td></tr>
  <tr><td>I finish what I start.</td>
      <td><input type=radio name=B1 value=1></td><td><input type=radio name=B1 value=2></td>
      <td><input type=radio name=B1 value=3></td><td><input type=radio name=B1 value=4></td>
      <td><input type=radio name=B1 value=5></td></tr>
  <tr><td>I enjoy crowds.</td>
      <td><input type=radio name=B2 value=1></td><td><input type=radio name=B2 value=2></td>
      <td><input type=radio name=B2 value=3></td><td><input type=radio name=B2 value=4></td>
      <td><input type=radio name=B2 value=5></td></tr>
</table></form>
"""

LABELLED_OPTIONS_UNDER_A_LEGEND = """
<form><fieldset><legend>How do you usually decide?</legend>
  <label><input type=radio name=C1 value=a> By reasons</label>
  <label><input type=radio name=C1 value=b> By feel</label>
</fieldset></form>
"""


def _observed(html: str) -> dict:
    playwright = pytest.importorskip("playwright.async_api")

    async def look() -> dict:
        async with playwright.async_playwright() as running:
            try:
                browser = await running.chromium.launch(headless=True)
            except Exception as why:  # noqa: BLE001 - no browser here is a skip, not a failure
                pytest.skip(f"no headless browser here: {why}")
            try:
                page = await browser.new_page()
                await page.set_content(html)
                return await page.evaluate(PhantomBrowser._OBSERVE_SCRIPT, 120)
            finally:
                await browser.close()

    return asyncio.run(look())


def _asks(observation: dict, group: str) -> tuple[str, str]:
    options = [one for one in observation["elements"] if one.get("group") == group]
    assert options, f"no options seen for {group}"
    return options[0].get("asks", ""), options[0].get("heading", "")


def test_a_scale_between_two_phrases_is_read_in_its_place():
    seen = _observed(A_SCALE_BETWEEN_TWO_PHRASES)
    asks, heading = _asks(seen, "A1")
    assert asks == "plans ahead [1] [2] [3] [4] [5] improvises"
    assert heading == ""
    assert SovereignBrowserSkill._asks_about_the_one_answering(seen), (
        "unlabelled options told apart by value are one scale across questions"
    )


def test_a_grid_carries_its_heading():
    seen = _observed(A_GRID_UNDER_A_HEADING)
    asks, heading = _asks(seen, "B2")
    assert asks == "I enjoy crowds. [1] [2] [3] [4] [5]"
    assert heading.split() == ["Never", "Sometimes", "Always"]
    assert SovereignBrowserSkill._asks_about_the_one_answering(seen)


def test_labelled_options_are_read_with_their_labels_and_their_question():
    seen = _observed(LABELLED_OPTIONS_UNDER_A_LEGEND)
    asks, _heading = _asks(seen, "C1")
    assert asks == "How do you usually decide? [a] By reasons [b] By feel"


def test_the_decision_sees_each_question_once_above_its_options():
    rendered = SovereignBrowserSkill._render_observation(
        {
            "url": "https://example.test/q",
            "title": "q",
            "text": "",
            "elements": [
                {"role": "radio", "name": "A1", "group": "A1", "value": str(v),
                 "asks": "plans ahead [1] [2] [3] [4] [5] improvises", "selector": f"#a{v}"}
                for v in range(1, 6)
            ],
        }
    )
    assert rendered.count("question A1 reads: plans ahead [1] [2] [3] [4] [5] improvises") == 1
    assert rendered.index("question A1 reads") < rendered.index("[0] radio")


def test_options_named_only_by_their_group_are_told_apart_by_value():
    """No browser: the shape the live page gave, sixty groups of five like this."""
    page = {
        "elements": [
            {"role": "radio", "name": f"Q{q}", "group": f"Q{q}", "value": str(v), "selector": f"#q{q}v{v}"}
            for q in range(1, 4)
            for v in range(1, 6)
        ]
    }
    assert SovereignBrowserSkill._asks_about_the_one_answering(page)
