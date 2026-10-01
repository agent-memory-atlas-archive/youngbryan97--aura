"""A question about what she remembers, answered from the record.

Candidates come from episodic memory, long-term memory and the turns still in
working memory. They are ranked by MEANING: the cosine between the question and
each remembered sentence, from the runtime's one shared sentence encoder.

Measured 30 September on twelve question-and-memory pairs set among each
other's memories and six same-topic near misses: ranked by meaning, the memory
that answers came first for 11 of 12. Ranked by the word scorer below, 1 of
12, because a length bonus let any sentence of 12 to 220 characters clear its
bar, so the direct answer quoted whichever memory scored top, often the wrong
one. The one miss by meaning is a paraphrase beaten by a near miss on the same
topic ("where I said I was going on holiday": "I took last year's holiday at
home" over "We're flying to Lisbon in October"), and it was also the match that
stood out most, so no margin can catch it.

So a memory is quoted back without the model only when the best by meaning is
also ANCHORED: it shares a distinctive word with the question, matched as a
word. On the twelve that quotes six correctly, quotes the holiday miss, and
leaves five to her cortex, which answers from the same memories ordered by
meaning and can see that flying to Lisbon is going on holiday. Without an
encoder the word scorer ranks and the same anchor decides.

What comes back is a remembered sentence with nothing added. A recalled
answer that has been smoothed is a confabulation with good manners.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

from core.conversation.word_markers import names_any, names_marker
from core.language.learned_matcher import cosine, embed_sentences
from core.state.aura_state import AuraState
from core.utils.intent_normalization import normalize_memory_intent_text

logger = logging.getLogger(__name__)



class _AnswersFromWhatSheRemembers:
    """Lifted whole from UnitaryResponsePhase; see response_generation_unitary.py."""

    @classmethod
    def _is_explicit_memory_recall_request(cls, objective: str) -> bool:
        lowered = normalize_memory_intent_text(cls._normalize_text(objective))
        if not lowered:
            return False
        # Strict markers: phrases that unambiguously ask for memory recall
        explicit_markers = (
            "what was the exact phrase",
            "what was the phrase",
            "what were the exact words",
            "what did i tell you to remember",
            "what did i mean when i said",
            "what do you remember i said",
            "do you remember when i",
            "do you remember what i",
            "what do you remember about",
            "can you recall",
            "told you to remember",
            "remember forever",
            "recall what i said",
            "recall what i told",
        )
        if names_any(lowered, explicit_markers):
            return True
        # Require the word "remember" or "recall" explicitly paired with a
        # recall-specific question form. Generic words like "before", "earlier"
        # are NOT sufficient on their own -- they appear in normal conversation
        # (e.g. "wait before I do, what do YOU want?").
        has_recall_verb = names_any(lowered, ("remember", "recall"))
        has_recall_question = any(
            token in lowered
            for token in (
                "what was",
                "what did i",
                "what do you remember",
                "exact phrase",
                "exact words",
            )
        )
        return has_recall_verb and has_recall_question

    @classmethod
    def _is_idle_introspection_request(cls, objective: str) -> bool:
        lowered = cls._normalize_text(objective).lower()
        if not lowered:
            return False
        explicit_markers = (
            "what have you been thinking",
            "what were you thinking",
            "while idle",
            "between my messages",
            "between messages",
            "during the pause",
            "when i was gone",
            "idle thought",
        )
        if names_any(lowered, explicit_markers):
            return True
        return names_any(lowered, ("thinking", "thought", "idle")) and names_any(
            lowered, ("between", "while", "during", "when i was gone")
        )

    @classmethod
    def _looks_like_meta_recall_query(cls, text: str) -> bool:
        lowered = normalize_memory_intent_text(cls._normalize_text(text))
        if not lowered or not lowered.endswith("?"):
            return False
        return any(
            marker in lowered
            for marker in (
                "what was the exact phrase",
                "what was the phrase",
                "what were the exact words",
                "what did i tell you",
                "what do you remember",
                "earlier today i told you",
                "remember forever",
                "what have you been thinking",
                "what were you thinking",
            )
        )

    @classmethod
    def _extract_user_utterance(cls, raw: Any) -> str:
        text = cls._normalize_text(raw)
        if not text:
            return ""

        text = re.sub(r"^\[[^\]]+\]\s*", "", text).strip()
        for prefix_pattern in (r"user said:\s*(.+)", r"context:\s*(.+)"):
            match = re.search(prefix_pattern, text, flags=re.IGNORECASE)
            if match:
                text = match.group(1).strip()
        text = re.split(
            r"\s*\|\s*(?:conversation_reply|assistant_reply|reply|response)\s*\|\s*",
            text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        text = re.split(r"\s*\|\s*action:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\s*\|\s*outcome:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = re.split(r"\s*→\s*", text, maxsplit=1)[0]
        return cls._normalize_text(text).strip(" \"'")

    @classmethod
    def _collect_memory_evidence_lines(
        cls,
        state: AuraState,
        episodic_matches: list[Any] | None = None,
        *,
        limit: int = 4,
    ) -> list[str]:
        # Imported here rather than at module level: the module these
        # came from imports this one to build the class. A call-time
        # import also still sees a test's patch of the original.
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
        )

        lines: list[str] = []
        seen: set[str] = set()

        for ep in episodic_matches or []:
            try:
                if hasattr(ep, "to_retrieval_text"):
                    evidence = cls._normalize_text(ep.to_retrieval_text(), 340)
                else:
                    evidence = cls._normalize_text(
                        getattr(ep, "full_description", "") or getattr(ep, "context", ""),
                        340,
                    )
            except _RESPONSE_RECOVERABLE_ERRORS as exc:
                logger.debug(
                    "an episodic match would not render, so it contributes no evidence (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                evidence = ""
            if evidence and evidence not in seen:
                seen.add(evidence)
                lines.append(evidence)

        for item in list(getattr(state.cognition, "long_term_memory", []) or []):
            evidence = cls._normalize_text(item, 340)
            if evidence and evidence not in seen:
                seen.add(evidence)
                lines.append(evidence)

        return lines[:limit]

    @classmethod
    def _collect_recent_turn_evidence_lines(
        cls,
        state: AuraState,
        *,
        limit: int = 4,
    ) -> list[str]:
        lines: list[str] = []
        seen: set[str] = set()

        for item in reversed(list(getattr(state.cognition, "working_memory", []) or [])[-12:]):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "").strip().lower()
            content = cls._normalize_text(item.get("content", ""), 260)
            if not content:
                continue
            if role == "assistant":
                line = f"Aura said: {content}"
            elif role == "user":
                line = f"User said: {content}"
            else:
                line = content
            if line not in seen:
                seen.add(line)
                lines.append(line)
            if len(lines) >= limit:
                break

        return lines[:limit]

    @staticmethod
    async def _direct_episodic_matches(objective: str, limit: int = 3) -> list[Any]:
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
            _record_response_degradation,
            logger,
        )

        try:
            from core.container import ServiceContainer

            episodic = ServiceContainer.get("episodic_memory", default=None)
            if not episodic:
                return []
            if hasattr(episodic, "recall_similar_async"):
                matches = await episodic.recall_similar_async(objective, limit=limit)
            elif hasattr(episodic, "recall_similar"):
                matches = await asyncio.to_thread(episodic.recall_similar, objective, limit)
            else:
                return []
            return list(matches or [])
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: direct episodic grounding failed: %s",
                action="returned no direct episodic matches after direct recall failed",
            )
            logger.debug("UnitaryResponse: direct episodic grounding failed: %s", exc)
            return []

    @staticmethod
    async def _recent_episodic_matches(limit: int = 80) -> list[Any]:
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
            _record_response_degradation,
            logger,
        )

        try:
            from core.container import ServiceContainer

            episodic = ServiceContainer.get("episodic_memory", default=None)
            if not episodic:
                return []
            if hasattr(episodic, "recall_recent_async"):
                matches = await episodic.recall_recent_async(limit=limit)
            elif hasattr(episodic, "recall_recent"):
                matches = await asyncio.to_thread(episodic.recall_recent, limit)
            else:
                return []
            return list(matches or [])
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: recent episodic recall failed: %s",
                action="returned no recent episodic matches after recall failed",
            )
            logger.debug("UnitaryResponse: recent episodic recall failed: %s", exc)
            return []

    @classmethod
    def _token_distinctiveness(cls, token: str) -> float:
        """How much a match on this token should count, in [0, 4].

        Replaces a length floor that scored "something" and ignored "fox".
        Distinctiveness comes from three things a stopword list cannot fake:

          * digits and punctuation-bearing tokens are almost always specific
            ("3:14", "v2", "412") — these are the ones a person quotes back;
          * a token absent from the stopword list carries content;
          * very short tokens are ambiguous ONLY when they are also common,
            so shortness alone is not a penalty.

        Bounded above so no single token can dominate the score the way the
        hardcoded +4.0 did.
        """
        word = str(token or "").strip().lower()
        if not word or word in cls._RECALL_STOPWORDS:
            return 0.0
        has_digit = any(character.isdigit() for character in word)
        has_separator = any(character in ":._-/" for character in word)
        if has_digit or has_separator:
            # A number or a structured token is the thing people quote back
            # verbatim, and matching one is strong evidence.
            return 2.5
        if len(word) < 3:
            # One and two-letter tokens are function words or fragments.
            return 0.0
        # Every other content word counts the SAME.
        #
        # The first version of this graded by length — 1.5 at eight
        # characters, 1.0 at five, 0.75 at three — and a test comparing "fox"
        # against "otter" caught it: the two scored differently for the same
        # sentence, which is the very asymmetry the hardcoded bonuses
        # created. Grading by length also contradicts the argument directly
        # above it, that distinctiveness is not length.
        #
        # Without a corpus there is no honest basis for a gradient, and an
        # invented one is a magic number that quietly decides which memories
        # surface. Equal weight is the claim the evidence supports.
        return 1.0

    @classmethod
    def _score_memory_candidate(cls, candidate: str, objective: str) -> float:
        text = cls._normalize_text(candidate)
        lowered = text.lower()
        objective_lower = normalize_memory_intent_text(cls._normalize_text(objective))
        score = 0.0

        if 12 <= len(text) <= 220:
            score += 2.0
        elif len(text) <= 320:
            score += 0.5
        else:
            score -= min(5.0, (len(text) - 320) / 80.0)

        if "remember" in lowered:
            score += 3.0
        if "forever" in lowered:
            score += 3.0
        if "exact phrase" in lowered or "phrase" in lowered:
            score += 1.5
        # The three literal boosts that used to live here — "fox" +4.0,
        # "3:14" +2.5, "bryan" +1.5 — are gone.
        #
        # They were not arbitrary: they were a patch over a real defect
        # immediately below. The general overlap rule required
        # ``len(token) > 3``, so "fox" scored NOTHING through the general
        # path, and someone made the demo work by naming it. The cost was
        # that the very examples used to show memory working were the ones
        # the scorer privileged, so those demos could not be read as
        # evidence about general retrieval at all.
        #
        # The fix is to the cause. A token's worth is its DISTINCTIVENESS,
        # not its length: "fox" and "3:14" are short and highly specific,
        # while "about" and "something" are longer and carry nothing. A
        # length floor gets that exactly backwards.
        objective_tokens = set(re.findall(r"[a-z0-9:]+", objective_lower))
        for token in objective_tokens:
            if token not in lowered:
                continue
            score += cls._token_distinctiveness(token)

        if lowered.endswith("?"):
            score -= 2.0
        if cls._looks_like_meta_recall_query(text):
            score -= 4.0

        bad_markers = (
            "silent auto-fix",
            "traceback",
            "task exception",
            "background cognitive state",
            "background_consolidation",
            "return only the json",
            "diagnosing a recurring bug",
            "cognitive baseline tick",
            "future: <task finished",
        )
        if any(marker in lowered for marker in bad_markers):
            score -= 8.0

        return score

    #: Verbs that name the act of remembering. A question asks "what did I tell
    #: you to remember" of a sentence that said "remember this", so the shared
    #: verb points at that sentence even though recall questions all use it.
    _MEMORY_ACTS: tuple[str, ...] = ("remember", "forget")

    #: Text that is the runtime talking to itself, never something said.
    _NOT_SAID: tuple[str, ...] = (
        "silent auto-fix",
        "traceback",
        "task exception",
        "background cognitive state",
        "background_consolidation",
        "return only the json",
        "cognitive baseline tick",
        "future: <task finished",
    )

    @classmethod
    def _recall_candidates(
        cls,
        objective: str,
        state: AuraState,
        episodic_matches: list[Any] | None = None,
    ) -> list[tuple[str, str]]:
        """Every remembered sentence that could answer, with who said it."""
        candidates: list[tuple[str, str]] = []
        objective_norm = normalize_memory_intent_text(cls._normalize_text(objective)).rstrip("?")
        for ep in episodic_matches or []:
            for raw in (
                getattr(ep, "context", ""),
                getattr(ep, "description", ""),
                getattr(ep, "full_description", ""),
            ):
                utterance = cls._extract_user_utterance(raw)
                if utterance:
                    candidates.append(("user", utterance))
        for item in list(getattr(state.cognition, "long_term_memory", []) or []):
            utterance = cls._extract_user_utterance(item)
            if utterance:
                candidates.append(("user", utterance))
        for item in reversed(list(getattr(state.cognition, "working_memory", []) or [])[-24:]):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = cls._normalize_text(item.get("content", ""), 500)
            if content:
                candidates.append((role, content))

        kept: list[tuple[str, str]] = []
        seen: set[str] = set()
        for role, candidate in candidates:
            normalized = cls._normalize_text(candidate).lower().rstrip("?")
            if not normalized or len(normalized) < 8 or normalized == objective_norm:
                continue
            if cls._looks_like_meta_recall_query(candidate):
                continue
            if any(marker in normalized for marker in cls._NOT_SAID):
                continue
            key = f"{role}:{normalized}"
            if key not in seen:
                seen.add(key)
                kept.append((role, candidate))
        return kept

    @classmethod
    def _speaker_asked_about(cls, objective: str) -> str:
        """"assistant" or "user" when the question names who said it, else ""."""
        objective_norm = normalize_memory_intent_text(cls._normalize_text(objective))
        if any(
            marker in objective_norm
            for marker in (
                "what did you say",
                "what were your exact words",
                "what was your answer",
                "what did your reply",
                "what did you tell me",
            )
        ):
            return "assistant"
        if any(
            marker in objective_norm
            for marker in (
                "what did i say",
                "what did i tell",
                "what was my",
                "what were my exact words",
                "what do you remember i said",
                "do you remember what i",
            )
        ):
            return "user"
        return ""

    @classmethod
    def _anchor(cls, candidate: str, objective: str) -> float:
        """How much the question's own words point at this sentence, as words."""
        objective_lower = normalize_memory_intent_text(cls._normalize_text(objective))
        anchor = 0.0
        for token in set(re.findall(r"[a-z0-9:]+", objective_lower)):
            if names_marker(candidate, token):
                anchor += cls._token_distinctiveness(token)
        for act in cls._MEMORY_ACTS:
            if names_marker(objective_lower, act) and names_marker(candidate, act):
                anchor += 1.0
        return anchor

    @staticmethod
    def _meaning_of(objective: str, texts: list[str]) -> dict[str, float]:
        """Cosine from the question to each text, or {} with no encoder.

        Synchronous and slow: one encoder pass per sentence. Callers on the
        event loop run it through off_the_loop.
        """
        unique = list(dict.fromkeys(text for text in texts if text))
        if not unique:
            return {}
        vectors = embed_sentences([objective, *unique])
        if len(vectors) != len(unique) + 1:
            return {}
        return {text: cosine(vectors[0], vector) for text, vector in zip(unique, vectors[1:], strict=True)}

    async def _recall_by_meaning(
        self,
        objective: str,
        state: AuraState,
        episodic_matches: list[Any],
    ) -> tuple[str | None, list[Any]]:
        """The direct answer if one is warranted, and the episodes by meaning.

        The episodes come back reordered so the evidence handed to her cortex,
        which takes the first few, is the closest in meaning rather than the
        first retrieved.
        """
        from core.runtime.executors import off_the_loop

        candidates = self._recall_candidates(objective, state, episodic_matches)
        meaning = await off_the_loop(self._meaning_of, objective, [text for _role, text in candidates])
        if meaning:
            def _closeness(ep: Any) -> float:
                said = (
                    self._extract_user_utterance(getattr(ep, field, ""))
                    for field in ("context", "description", "full_description")
                )
                return max((meaning.get(text, -1.0) for text in said), default=-1.0)

            episodic_matches = sorted(episodic_matches, key=_closeness, reverse=True)
        answer = self._compose_memory_recall_answer(
            objective, state, episodic_matches, meaning=meaning, candidates=candidates
        )
        return answer, episodic_matches

    @classmethod
    def _compose_memory_recall_answer(
        cls,
        objective: str,
        state: AuraState,
        episodic_matches: list[Any] | None = None,
        *,
        meaning: dict[str, float] | None = None,
        candidates: list[tuple[str, str]] | None = None,
    ) -> str | None:
        """A remembered sentence quoted back, or None to let her cortex answer."""
        pool = candidates if candidates is not None else cls._recall_candidates(
            objective, state, episodic_matches
        )
        speaker = cls._speaker_asked_about(objective)
        if speaker:
            pool = [(role, text) for role, text in pool if role == speaker]
        if not pool:
            return None
        if meaning:
            chosen_role, chosen = max(pool, key=lambda item: meaning.get(item[1], -1.0))
        else:
            chosen_role, chosen = max(
                pool, key=lambda item: cls._score_memory_candidate(item[1], objective)
            )
        if cls._anchor(chosen, objective) <= 0.0:
            return None
        objective_norm = normalize_memory_intent_text(cls._normalize_text(objective))
        if any(marker in objective_norm for marker in ("exact phrase", "exact words", "exact wording")):
            return f'I said: "{chosen}"' if chosen_role == "assistant" else f'You told me: "{chosen}"'
        if chosen_role == "assistant":
            return f'I remember saying: "{chosen}"'
        return f'I remember you saying: "{chosen}"'

    @classmethod
    def _build_idle_trace_text(cls, state: AuraState) -> str:
        from .response_generation_unitary import (
            _RESPONSE_RECOVERABLE_ERRORS,
            _record_response_degradation,
            logger,
        )

        parts: list[str] = []
        try:
            from core.consciousness.stream_of_being import get_stream

            stream = get_stream()
            if hasattr(stream, "get_between_moments_text"):
                between = cls._normalize_text(stream.get_between_moments_text(), 320)
                if between and "I was here." not in between:
                    parts.append(between)
            if hasattr(stream, "get_status"):
                status = stream.get_status() or {}
                current = status.get("current_moment", {}) or {}
                focus = cls._normalize_text(current.get("focus"), 120)
                emotion = cls._normalize_text(current.get("emotion"), 60)
                arc = cls._normalize_text(status.get("arc_emotion"), 60)
                if focus:
                    parts.append(f"Current focus: {focus}")
                if emotion or arc:
                    parts.append(f"Emotional arc: {arc or emotion}")
        except _RESPONSE_RECOVERABLE_ERRORS as exc:
            _record_response_degradation(
                exc,
                "UnitaryResponse: idle trace unavailable: %s",
                action="continued idle introspection reply without stream-of-being trace",
            )
            logger.debug("UnitaryResponse: idle trace unavailable: %s", exc)

        pending: list[str] = []
        for item in list(getattr(state.cognition, "pending_initiatives", []) or [])[:2]:
            if not isinstance(item, dict):
                continue
            goal = cls._normalize_text(
                item.get("goal") or item.get("description") or item.get("type"), 100
            )
            if goal:
                pending.append(goal)
        if pending:
            parts.append(f"Pending initiatives: {', '.join(pending)}")

        return " ".join(part for part in parts if part).strip()

    @classmethod
    def _recent_assistant_claim(cls, state: AuraState, limit: int = 6) -> str:
        for item in reversed(list(getattr(state.cognition, "working_memory", []) or [])[-limit:]):
            if not isinstance(item, dict):
                continue
            if str(item.get("role", "") or "").strip().lower() != "assistant":
                continue
            content = cls._normalize_text(item.get("content", ""), 260)
            if content:
                return content
        return ""
