#!/usr/bin/env python3
"""Read a questionnaire the way she does, and say what came back.

Her page faculties without her cortex: arrive, observe, find the way in,
observe again, and report whether the page was taken for a scale instrument
and how each item reads. The point is that the reading is separable from the
answering — when a live demo fails it is worth knowing which half broke.

    python tools/check_a_questionnaire_reads.py https://openpsychometrics.org/tests/OEJTS/

Live and networked on purpose. Nothing here decides anything or answers
anything; it looks.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

#: What a page calls the way in. Read from the page's own words, not a list of
#: sites: every intake flow has one and they all say some version of this.
_WAYS_IN = ("start", "begin", "take the test", "next", "continue", "go")


def _grouped(observation: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for element in observation.get("elements") or []:
        name = str(element.get("group") or "")
        if name:
            groups.setdefault(name, []).append(element)
    return groups


def _say(observation: dict[str, Any], *, label: str) -> bool:
    from core.skills.sovereign_browser_understanding import _UnderstandsThePage

    groups = _grouped(observation)
    scale = bool(_UnderstandsThePage._asks_about_the_one_answering(observation))
    print(f"\n=== {label} ===")
    print("url:", observation.get("url"))
    print("elements:", len(observation.get("elements") or []), "groups:", len(groups))
    print("a scale instrument, so the questions are about her:", scale)
    for name, offered in list(groups.items())[:3]:
        first = offered[0]
        print(f"-- {name}")
        print("   asks:", str(first.get("asks") or "")[:160] or "(nothing — the item did not reach the decision)")
        if first.get("heading"):
            print("   heading:", str(first.get("heading"))[:120])
        print("   options:", [str(one.get("value") or one.get("name") or "")[:10] for one in offered][:8])
    return scale


def _the_way_in(observation: dict[str, Any]) -> dict[str, Any] | None:
    for element in observation.get("elements") or []:
        said = str(element.get("name") or "").strip().lower()
        if said in _WAYS_IN and element.get("selector"):
            return dict(element)
    return None


async def main(url: str) -> int:
    from core.capabilities.phantom_browser import PhantomBrowser

    browser = PhantomBrowser()
    arrived: dict[str, Any] = {}
    inside: dict[str, Any] = {}
    try:
        if not await browser.ensure_ready():
            print("the browser did not come up", file=sys.stderr)
            return 2
        if not await browser.browse(url, principal="questionnaire-check"):
            print(f"did not arrive at {url}", file=sys.stderr)
            return 2
        arrived = await browser.observe(principal="questionnaire-check", max_elements=300)
        _say(arrived, label="arrived")
        way_in = _the_way_in(arrived)
        if way_in is not None and not _grouped(arrived):
            print("\nthe way in:", way_in.get("name"), "→", way_in.get("selector"))
            await browser.page.click(str(way_in["selector"]))
            await browser.page.wait_for_load_state("domcontentloaded")
            inside = await browser.observe(principal="questionnaire-check", max_elements=300)
    finally:
        await browser.close()
    if inside:
        scale = _say(inside, label="inside")
        return 0 if scale else 1
    return 0 if _grouped(arrived) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    sys.exit(asyncio.run(main(parser.parse_args().url)))
