"""A long questionnaire page is seen whole, and answering its lower half counts.

Rehearsed on 27 Sep against the real OEJTS page one, with a stand-in deciding:
32 questions of five choices and a Next button, 161 controls. The observation
stopped at 120, so the last eight questions and Next were never seen; and the
progress signature read the first 60, so answering questions 13 to 32 looked
like no progress, and the run ended after two such rounds with Next unpressed.
And a look taken while a page was still loading went back to the page before.
"""

from __future__ import annotations

import asyncio

import pytest

from core.capabilities.phantom_browser import PhantomBrowser
from core.skills.sovereign_browser import SovereignBrowserSkill

QUESTIONS = 40


def _a_long_scale() -> str:
    rows = "".join(
        f"<tr><td>left {q}</td>"
        + "".join(f"<td><input type=radio name=Q{q} value={v}></td>" for v in range(1, 6))
        + f"<td>right {q}</td></tr>"
        for q in range(1, QUESTIONS + 1)
    )
    return f"<form><table>{rows}</table><input type=submit value=Next></form>"


def _observed_uncapped(html: str) -> dict:
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
                return await page.evaluate(PhantomBrowser._OBSERVE_SCRIPT, 0)
            finally:
                await browser.close()

    return asyncio.run(look())


def test_every_control_on_a_long_page_is_seen():
    seen = _observed_uncapped(_a_long_scale())
    groups = {one.get("group") for one in seen["elements"] if one.get("group")}
    assert len(groups) == QUESTIONS
    assert any(str(one.get("name")) == "Next" for one in seen["elements"]), "the way on is at the bottom"


def test_an_answer_at_the_bottom_of_the_page_is_progress():
    elements = [
        {"role": "radio", "name": f"Q{q}", "group": f"Q{q}", "value": str(v), "checked": False}
        for q in range(1, QUESTIONS + 1)
        for v in range(1, 6)
    ]
    before = {"url": "u", "elements": elements}
    after_elements = [dict(one) for one in elements]
    after_elements[-1]["checked"] = True
    after = {"url": "u", "elements": after_elements}
    assert SovereignBrowserSkill._observation_signature(before) != SovereignBrowserSkill._observation_signature(after)


def test_a_page_still_loading_is_waited_for_not_left():
    looked: list[str] = []

    class Page:
        async def wait_for_load_state(self, state, timeout=None):
            looked.append(f"waited:{state}")

    class Browser:
        page = Page()

        async def observe(self, principal=""):
            looked.append("observed")
            return {"url": "https://x.test/2", "elements": [{"role": "button", "name": "Next"}]}

    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    seen = asyncio.run(skill._look_again_once_loaded(Browser()))
    assert looked == ["waited:domcontentloaded", "observed"]
    assert seen["url"] == "https://x.test/2"
