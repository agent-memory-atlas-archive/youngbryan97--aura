"""A browser start is waited for as long as its own launch attempts may take.

LIVE 27 Sep, on a host in Low Power Mode: the pursuit that was to take a
sixty-item personality test waited a flat 30 seconds for its browser to start,
the start took longer, and a run sized in hours ended before its first page
with "Browser operation timed out: pursue". Each Playwright launch attempt has
its own 30-second bound, and there may be several.
"""

from __future__ import annotations

import asyncio

from core.capabilities.phantom_browser import PhantomBrowser
from core.skills.sovereign_browser import SovereignBrowserSkill


def test_the_wait_covers_the_driver_and_every_attempt(monkeypatch):
    browser = PhantomBrowser(visible=False, browser_type="chromium", principal="test")
    monkeypatch.setattr(browser, "_chromium_launch_candidates", lambda: [("bundled", ""), ("system", "/x")])
    assert browser.startup_bound_s() == browser.LAUNCH_TIMEOUT_S * (1 + 1 + 2)
    other = PhantomBrowser(visible=False, browser_type="firefox", principal="test")
    monkeypatch.setattr(other, "_chromium_launch_candidates", lambda: [("bundled", "")])
    assert other.startup_bound_s() == other.LAUNCH_TIMEOUT_S * (1 + 2 + 1)


def test_a_start_slower_than_one_attempt_is_still_waited_for(monkeypatch):
    waited: dict[str, float] = {}

    async def fake_wait_for(awaitable, timeout):
        waited["timeout"] = timeout
        return await awaitable

    async def ready(self):
        return True

    monkeypatch.setattr(PhantomBrowser, "ensure_ready", ready)
    monkeypatch.setattr(PhantomBrowser, "startup_bound_s", lambda self: 120.0)
    from core.skills import sovereign_browser

    monkeypatch.setattr(sovereign_browser.asyncio, "wait_for", fake_wait_for)
    skill = SovereignBrowserSkill.__new__(SovereignBrowserSkill)
    monkeypatch.setattr(skill, "_pick_browser_type", lambda preference: "chromium", raising=False)
    asyncio.run(skill._create_browser("chromium", visible=False))
    assert waited["timeout"] == 120.0
