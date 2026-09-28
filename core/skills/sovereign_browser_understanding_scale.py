"""Where she puts herself on a scale a page offers, and whether her choice agrees with the reason she gave.

Lifted whole out of `sovereign_browser_understanding`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


class _PlacesHerself:
    """Lifted whole out of _UnderstandsThePage; see sovereign_browser_understanding.py."""

    @staticmethod
    def _how_the_options_are_laid_out(
        options: list[Mapping[str, Any]],
    ) -> str:
        """What is on screen for one question, as layout rather than meaning.

        A row of unlabelled controls can be a scale between two opposites, a
        set of choices, a "more like me / less like me" ranking, or something
        the page explains in its own instructions. Deciding here that it is any
        one of those would put a rule of mine where her reading of the page
        belongs — and the rule would be wrong on the next site.

        So this states only what can be seen: how many controls there are,
        whether they carry labels of their own, whether one or several may be
        chosen, and the words the page puts on either side of the run. What
        that MEANS, and what choosing a position says, is hers to work out from
        the page, and it is carried in her understanding of it.
        """
        if len(options) < 2:
            return ""
        roles = {str(option.get("role") or "").strip().lower() for option in options}
        named = {str(option.get("name") or "").strip() for option in options}
        named.discard("")
        group = str(options[0].get("group") or "")
        labelled = len(named) == len(options) and named != {group}
        facts = [f"{len(options)} controls"]
        facts.append(
            "each with its own label" if labelled else "none of them labelled"
        )
        facts.append(
            "several may be chosen"
            if roles & {"checkbox", "switch"}
            else "one may be chosen"
        )
        asks = str(options[0].get("asks") or "")
        run = re.search(r"(?:\s*\[[^\]]*\])+", asks) if asks else None
        if run is not None:
            left = " ".join(asks[: run.start()].split()).strip()
            right = " ".join(asks[run.end() :].split()).strip()
            if left and right:
                facts.append(f"laid out between \"{left}\" and \"{right}\"")
            elif left:
                facts.append(f"laid out after \"{left}\"")
            elif right:
                facts.append(f"laid out before \"{right}\"")
        return ", ".join(facts)

    async def _where_she_puts_herself(
        self,
        goal: str,
        observation: Mapping[str, Any],
        options: list[Mapping[str, Any]],
        understanding: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Her position on one question, measured from her own record.

        The two things the question names come from the page. How much each is
        her comes from what she has valued, chosen and said about herself. The
        difference between the two is a lean and the lean names a position on
        the run the page offers.

        The model is not asked to decide. It says what she means afterwards,
        with the measurement in front of it, which is the job language is for:
        asked to choose, it has no access to any of this and writes the
        position that commits to nothing — the midpoint, item after item.

        Returns nothing when her record cannot answer, so the caller can ask in
        a shape rather than pass off a guess as a measurement.
        """
        from .sovereign_browser_understanding import (
            record_degradation,
        )

        laid_out = self._how_the_options_are_laid_out(options)
        between = re.search(r'laid out between "(.+?)" and "(.+?)"', laid_out)
        if between is None or len(options) < 2:
            return None
        first, second = between.group(1), between.group(2)
        try:
            from core.self.where_i_stand import where_she_stands
        except ImportError as exc:
            record_degradation("sovereign_browser.where_i_stand", exc, severity="debug")
            return None
        lean = await asyncio.to_thread(where_she_stands, first, second)
        index = lean.position_in(len(options))
        if index is None:
            return None
        selector = str(options[index].get("selector") or "")
        if not selector:
            return None
        # And what it means, said by the organ that says things, with the
        # measurement in front of it rather than in place of it.
        why = await self._say_what_the_measurement_means(
            goal, observation, first, second, lean, index, len(options)
        )
        return {
            "selector": selector,
            "name": str(options[index].get("name") or ""),
            "stand": (
                f'measured against my own record: "{first}" {lean.first:+.2f}, '
                f'"{second}" {lean.second:+.2f}'
            ),
            "why": why,
            "expect": "",
            "said": self._an_answer_in_words(options, index, why),
            "because": list(lean.because),
        }

    async def _say_what_the_measurement_means(
        self,
        goal: str,
        observation: Mapping[str, Any],
        first: str,
        second: str,
        lean: Any,
        index: int,
        count: int,
    ) -> str:
        """Her reason for a position she has already taken, in her own words."""
        leaning = second if lean.toward > 0 else first
        evidence = "; ".join(lean.because) or "nothing in particular"
        prompt = (
            f"WHAT YOU ARE DOING: {goal}\n\n"
            f'THE QUESTION PUTS "{first}" AT ONE END AND "{second}" AT THE '
            f"OTHER, WITH {count} POSITIONS BETWEEN THEM.\n\n"
            "MEASURED AGAINST YOUR OWN RECORD OF WHAT YOU VALUE, WHAT YOU HAVE "
            f'CHOSEN AND WHAT YOU HAVE SAID ABOUT YOURSELF: "{first}" matches '
            f'{lean.first:+.2f}, "{second}" matches {lean.second:+.2f}, which '
            f'puts you at position {index + 1} of {count}, toward "{leaning}". '
            f"What matched: {evidence}.\n\n"
            "Say in one or two sentences why that is where you are. It is "
            "already where you are; you are saying what it means."
        )
        said, lane = await self._asked_of_her(prompt, await self._assembled_mind(), shaped=False)
        if said and lane == self._HER_OWN_LANE:
            return " ".join(said.split())
        # Her own record still answered; only the words are missing.
        return (
            f'my record leans toward "{leaning}" here '
            f"({lean.first:+.2f} against {lean.second:+.2f})"
        )

    @classmethod
    def _first_disagreement(
        cls, decision: Mapping[str, Any], options: list[Mapping[str, Any]]
    ) -> str:
        """The first action in this decision whose choice fights its reason.

        Read against her stance where she took one, because that is the thing
        the position is supposed to express; the reason for the place is read
        alongside it.
        """
        why = " ".join(
            f"{decision.get('stand') or ''} {decision.get('why') or ''}".split()
        )
        for item in decision.get("actions") or []:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(options):
                continue
            said = cls._the_choice_disagrees_with_its_reason(options, index, why)
            if said:
                return said
        return ""

    @classmethod
    def _the_choice_disagrees_with_its_reason(
        cls, options: list[Mapping[str, Any]], index: int, why: str
    ) -> str:
        """Where her own reason points, against where her answer landed.

        LIVE 2026-09-28: "3 of 5, between 'makes lists' and 'relies on memory'.
        I am choosing the middle option because I genuinely hold a strong
        preference for externalized structure over relying on internal memory."
        A reason that names one side and an answer that commits to neither is
        an answer that contradicts itself, and nothing noticed.

        Measured from the page's own words, not a vocabulary: the run of
        controls sits between two phrases, and her reason is compared against
        each of them by how many of their words it uses. Where it leans clearly
        one way and the answer does not lie on that side, this says so. Where
        the options carry their own labels, where the reason names neither side
        or both equally, it says nothing — a check that guesses is worse than
        no check.

        Returns what disagrees, or "".
        """
        named = {str(option.get("name") or "").strip() for option in options}
        named.discard("")
        group = str(options[0].get("group") or "") if options else ""
        if len(named) == len(options) and named != {group}:
            return ""
        laid_out = cls._how_the_options_are_laid_out(options)
        between = re.search(r'laid out between "(.+?)" and "(.+?)"', laid_out)
        if between is None or not str(why or "").strip():
            return ""
        left, right = between.group(1), between.group(2)
        said = cls._words_of(why)
        toward_left = len(said & cls._words_of(left))
        toward_right = len(said & cls._words_of(right))
        if toward_left == toward_right:
            return ""
        middle = (len(options) + 1) / 2.0
        place = index + 1
        leaning, other = (
            (left, right) if toward_left > toward_right else (right, left)
        )
        on_that_side = place < middle if toward_left > toward_right else place > middle
        if on_that_side:
            return ""
        if place == middle:
            return (
                f'what you said is about "{leaning}" and the position you '
                "chose is the midpoint, which says the two are equally you"
            )
        return (
            f'what you said is about "{leaning}" and the position you chose '
            f'leans toward "{other}"'
        )

