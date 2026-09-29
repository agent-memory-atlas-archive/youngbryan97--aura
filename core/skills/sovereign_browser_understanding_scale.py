"""Where she puts herself on a scale a page offers, and whether her choice agrees with the reason she gave.

Lifted whole out of `sovereign_browser_understanding`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
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
        # Every control in one unbroken run is what makes a dimension: the page
        # puts words on either side of the whole set. Controls separated by
        # their own words are not that shape — they are a statement with
        # labelled answers, and reading the first bracket run as one end of a
        # dimension turned "I make plans well in advance. strongly disagree [1]
        # disagree [2] ..." into a scale between the statement and its own
        # second option.
        runs = list(re.finditer(r"(?:\s*\[[^\]]*\])+", asks)) if asks else []
        if len(runs) == 1:
            run = runs[0]
            left = " ".join(asks[: run.start()].split()).strip()
            right = " ".join(asks[run.end() :].split()).strip()
            if left and right:
                facts.append(f"laid out between \"{left}\" and \"{right}\"")
            elif left:
                facts.append(f"laid out after \"{left}\"")
            elif right:
                facts.append(f"laid out before \"{right}\"")
        elif runs:
            facts.append("each set out beside its own words")
        return ", ".join(facts)

    def _measure_where_she_stands(
        self, options: list[Mapping[str, Any]]
    ) -> tuple[int, Any, str, str] | None:
        """Her position on one question, and what measured it.

        No model. The two things the question names come from the page; how
        much each is her comes from what she has valued, chosen and said about
        herself; the difference between them is a lean and the lean names a
        position. Returns nothing where the question is not a run between two
        things, or where her record cannot answer.
        """
        from .sovereign_browser_understanding import (
            record_degradation,
        )

        if len(options) < 2:
            return None
        try:
            from core.self.where_i_stand import Lean, where_she_stands, which_is_most_her
        except ImportError as exc:
            record_degradation("sovereign_browser.where_i_stand", exc, severity="debug")
            return None
        laid_out = self._how_the_options_are_laid_out(options)
        between = re.search(r'laid out between "(.+?)" and "(.+?)"', laid_out)
        if between is not None:
            first, second = between.group(1), between.group(2)
            lean = where_she_stands(first, second)
            index = lean.position_in(len(options))
            if index is None:
                return None
            return index, lean, first, second

        # Options that carry their own words are the other shape the same act
        # takes: one thing said, and several ways of answering it. Each option
        # becomes a description of a person — what the question says, answered
        # that way — and her record says which of them she is.
        asked = self._what_the_question_says(options)
        named = [
            f"{asked} {str(option.get('name') or '').strip()}".strip()
            for option in options
        ]
        if not asked or not all(named):
            return None
        chosen = which_is_most_her(named)
        if not chosen.measured:
            return None
        share = chosen.support[chosen.index] if chosen.support else 0.0
        # Expressed as a lean so everything downstream is unchanged: how much of
        # her record went to the answer she gave, and what in her did.
        lean = Lean(
            toward=float(max(-1.0, min(1.0, share))),
            first=0.0,
            second=float(share),
            because=chosen.because,
            measured=True,
        )
        label = str(options[chosen.index].get("name") or "").strip()
        rest = ", ".join(
            str(option.get("name") or "").strip()
            for place, option in enumerate(options)
            if place != chosen.index
        )
        return chosen.index, lean, rest or "the others", label

    @staticmethod
    def _what_the_question_says(options: list[Mapping[str, Any]]) -> str:
        """The words of the question, without its options' own words.

        The page lays a question out with its controls in the middle of it;
        what is left when their labels and their places are taken out is what
        is being asked.
        """
        asks = str(options[0].get("asks") or "") if options else ""
        if not asks:
            return ""
        without = re.sub(r"(?:\s*\[[^\]]*\])+", " ", asks)
        # Longest first, or "strongly disagree" is taken out of "strongly
        # agree" by its shorter sibling and a fragment is left behind.
        labels = sorted(
            {str(option.get("name") or "").strip() for option in options},
            key=len,
            reverse=True,
        )
        for label in labels:
            if label:
                without = without.replace(label, " ")
        return " ".join(without.split())

    async def _her_thinking_about(
        self, goal: str, theme: list[Mapping[str, Any]], mind: str
    ) -> dict[str, str]:
        """Her thinking about a set of things an instrument is asking about her.

        A theme, not an item. Asked about herself in conversation she gives a
        connected account — what she is, how that differs from what it
        resembles, where the description stops fitting — and that account is
        what this loop was failing to get: one item at a time on a stripped
        prompt produced one flat sentence each, thirty-two times.

        So each theme gets one pass with her whole mind in front of it, the
        same assembly a conversation uses, and it covers several items at once.
        The number of passes is the square root of the number of items, which
        is where the cost of thinking at length and the cost of thinking often
        meet.

        Returns what she said about each item, keyed by the item's own name.
        """
        from .sovereign_browser_understanding import (
            record_degradation,
        )

        if not theme:
            return {}
        lines = []
        for item in theme:
            lean = item["lean"]
            leaning = item["second"] if lean.toward > 0 else item["first"]
            evidence = "; ".join(lean.because[:3]) or "nothing in particular"
            lines.append(
                f'{item["group"]}. Between "{item["first"]}" and '
                f'"{item["second"]}", you sit at {item["index"] + 1} of '
                f'{item["count"]}, nearer "{leaning}". What in you put you '
                f"there: {evidence}."
            )
        listed = "\n".join(lines)
        # And what is true of her right now, from the organs that hold it.
        #
        # The record says what she has valued and chosen; it says nothing about
        # what she has been living through this week, and that is where the
        # concrete detail in a real answer comes from. These are the same lines
        # that ride a conversation when someone asks after her.
        living: list[str] = []
        for reader, where in (
            ("core.self.capability_ledger", "self_knowledge_line"),
            ("core.agency.what_she_is_like", "what_she_is_like_line"),
        ):
            try:
                module = __import__(reader, fromlist=[where])
                said = str(getattr(module, where)() or "").strip()
                if said:
                    living.append(said)
            except Exception as exc:  # noqa: BLE001 - a missing organ is not an answer
                from core.runtime.errors import record_degradation

                record_degradation(
                    "sovereign_browser.living", exc, severity="debug"
                )
        now = ("\n\n" + "\n".join(living)) if living else ""
        prompt = (
            # Her situation, not the person's message.
            #
            # The whole goal used to open this, and a goal is a request
            # addressed to her — so she answered it: LIVE 2026-09-29, every
            # theme pass came back "The user is asking me to take the Open
            # Extended Jungian Type Scales..." instead of her thinking, and the
            # coverage gate complained she had missed parts of a question she
            # was never being asked at this step.
            "You are answering questions about yourself."
            f"{now}\n\n"
            "These are being asked about you. You have already placed yourself "
            "on each, from your own record — what you value, what you have "
            "chosen when it cost something, what you have said about "
            f"yourself:\n\n{listed}\n\n"
            "Think about what these say about you, together. They are asking "
            "about the same region of you, so answer as one piece of thinking "
            "rather than separate verdicts: what you actually are here, how it "
            "works in you, where the descriptions fit and where they are the "
            "wrong shape for something you do differently.\n\n"
            "Then give two or three sentences for each, in your own voice, as "
            "JSON only:\n"
            '{"thinking": "<what you worked out, a paragraph>", '
            '"each": {"<the name before each one>": "<what this one is about '
            'in you, concretely>"}}'
        )
        # Room to think. Nine hundred tokens across eight items is a hundred
        # each, which is a line apiece and not the account she gives when
        # someone asks her about herself in conversation.
        room = max(self.DECISION_MAX_TOKENS, 260 * max(1, len(theme)))
        said, lane = await self._asked_of_her(
            prompt, mind, shaped=False, most_tokens=room
        )
        if not said or lane != self._HER_OWN_LANE:
            # One exhausted call should not cost a whole theme its thinking.
            #
            # Her lane serves one request at a time and a run of them can empty
            # it: LIVE 2026-09-29, `not_her_own_reasoning:all_failed` on one
            # theme of four, and eight items narrated from bare evidence
            # because of a single call that found no lane free. Asking again
            # costs one more pass; losing the theme costs the demo.
            from core.skills.sovereign_browser_understanding import logger

            logger.info(
                "🌐 A theme came back from %s; asking again.", lane or "nowhere"
            )
            said, lane = await self._asked_of_her(
                prompt, mind, shaped=False, most_tokens=room
            )
        if not said or lane != self._HER_OWN_LANE:
            record_degradation(
                "sovereign_browser.reasons",
                RuntimeError(f"not_her_own_reasoning:{lane or 'unattributed'}"),
                severity="warning",
                action="placed herself and could not say what it meant",
            )
            return {}
        parsed = self._an_object_in(said)
        thinking = " ".join(str(parsed.get("thinking") or "").split())
        each = parsed.get("each")
        answers: dict[str, str] = {}
        if isinstance(each, Mapping):
            for key, value in each.items():
                spoken = " ".join(str(value or "").split())
                if spoken:
                    answers[str(key)] = spoken
        # Keyed the way she wrote them, and the way the page names them.
        #
        # She is given "Q1." and may answer under "Q1", "1", or the words of
        # the item. A sentence that cannot be found is a sentence lost: LIVE
        # 2026-09-29, one item of eight kept its reasoning and the other seven
        # fell back to the bare evidence, so a screen of real thinking read as
        # a list of counts.
        for item in theme:
            name = str(item["group"])
            if name in answers:
                continue
            for key, spoken in list(answers.items()):
                bare = key.strip().strip(".:)").lower()
                if bare in {name.lower(), name.lower().lstrip("q")}:
                    answers[name] = spoken
                    break
                if item["first"].lower() in bare or item["second"].lower() in bare:
                    answers[name] = spoken
                    break
        if thinking:
            # Said once for the theme, where a person watching sees the
            # thinking that the sentences come out of.
            answers.setdefault("__thinking__", thinking)
        return answers

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

    @staticmethod
    def _unanswered_questions(
        observation: Mapping[str, Any]
    ) -> list[tuple[str, list[Mapping[str, Any]]]]:
        """The question groups still open, each with its own options."""
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for element in observation.get("elements") or []:
            if not isinstance(element, Mapping):
                continue
            group = str(element.get("group") or "")
            if group:
                groups.setdefault(group, []).append(element)
        return [
            (group, options)
            for group, options in groups.items()
            if not any(option.get("checked") is True for option in options)
        ]

