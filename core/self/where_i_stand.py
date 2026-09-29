"""Where she stands on a described dimension, measured from her own record.

A graded question names two things and asks which is more her. Handing that to a
language model and taking the number it writes is not her answering: the model
has no access to what she has valued or chosen, so it writes a plausible
sentence and picks the position that commits to nothing. Measured live on
2026-09-28, that position was the midpoint on item after item, under reasons
that named a strong preference.

So the position is measured. Her record already holds what she is like — the
values she holds with the weights they carry, what she has said about herself in
her own words, and what she actually chose when she had options — and each side
of a dimension is a description that either matches that record or does not. The
difference between the two matches is a lean, and a lean maps onto the positions
the page offers.

The model is not asked to decide. It says what she means afterwards, with the
measurement in front of it, which is the job language is for.

Nothing here knows what any particular instrument measures: the two sides come
from the page and the record comes from her.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_access import optional_service

logger = logging.getLogger(__name__)

__all__ = [
    "Choice",
    "Lean",
    "Piece",
    "her_record",
    "how_much_it_is_her",
    "against_the_rest",
    "themes_among",
    "where_she_stands",
    "which_is_most_her",
]


@dataclass(frozen=True)
class Piece:
    """One thing her record says about her, and how much it counts.

    ``said`` is the short form the measure compares against. ``about_her`` is
    the same thing as a person would put it, which is what she is handed when
    she is asked why a position is true of her. Nobody explains a personality
    answer by quoting a coefficient about themselves; they talk about what they
    value, what they keep doing, and what they have said before.
    """

    said: str
    weight: float = 1.0
    source: str = ""
    about_her: str = ""


@dataclass(frozen=True)
class Choice:
    """Which of several descriptions of a person her record supports.

    The sibling of a lean, for a question that offers labelled answers rather
    than a run between two ends: one statement and several ways of answering
    it, or a set of options that are not a line at all. The act is the same —
    place yourself among what is offered — and so is the evidence.
    """

    #: Which description, as an index into the ones given.
    index: int
    #: The share of her record that supported each, in the same order.
    support: tuple[float, ...] = field(default_factory=tuple)
    #: What in her supported the one she landed on, most telling first.
    because: tuple[str, ...] = field(default_factory=tuple)
    #: False when her record had nothing to say about any of them.
    measured: bool = False


@dataclass(frozen=True)
class Lean:
    """How far toward one side of a dimension her record puts her."""

    #: -1.0 entirely the first side, +1.0 entirely the second, 0.0 neither more.
    toward: float
    #: The match each side got, so the number can be argued with.
    first: float
    second: float
    #: What in her record matched, most first.
    because: tuple[str, ...] = field(default_factory=tuple)
    #: False when her record had nothing to say, and the caller must not
    #: mistake "no evidence" for "equally both".
    measured: bool = False
    #: How much closer her record is to one side than the other, before
    #: anything is made of it. ``toward`` says how consistently she leans;
    #: this says by how much, and it is what lets one question be compared
    #: with another asked of the same record.
    gap: float = 0.0

    def position_in(self, count: int) -> int | None:
        """Which of ``count`` positions this lean puts her at, 0-based.

        The run is a line from one side to the other, so a lean of -1 is the
        first position, +1 the last, and the rest fall where they land. An
        unmeasured lean names no position at all.
        """
        if not self.measured or count < 2:
            return None
        place = (self.toward + 1.0) / 2.0 * (count - 1)
        return max(0, min(count - 1, int(round(place))))


def _ordinal(place: int) -> str:
    """"st", "nd", "rd" or "th" for a small place in a ranking."""
    return {1: "st", 2: "nd", 3: "rd"}.get(place, "th")


def her_record() -> list[Piece]:
    """What she has to answer from: her values, her words, her choices.

    Every piece is something she produced. Nothing is invented here and
    nothing is inferred from the question being asked.
    """
    pieces: list[Piece] = []
    try:
        from core.agency.what_she_is_like import what_she_is_like

        portrait = what_she_is_like()
    except Exception as exc:  # noqa: BLE001 - a missing organ is not an answer
        record_degradation("where_i_stand", exc, severity="debug")
        portrait = None
    if portrait is not None:
        # The values she holds, with the weight she holds them at, and how she
        # chose when each was on offer. Both are hers and they are different
        # evidence: one is what she says she values, the other is what she did
        # when it cost something.
        ranked = sorted(
            (value for value in getattr(portrait, "values", ()) or ()),
            key=lambda value: float(getattr(value, "held", 0.0) or 0.0),
            reverse=True,
        )
        for place, value in enumerate(ranked):
            said = str(getattr(value, "value", "") or "").replace("_", " ")
            if not said:
                continue
            held = max(0.0, float(getattr(value, "held", 0.0) or 0.0))
            standing = (
                "the value I hold above every other"
                if place == 0
                else f"one of the things I hold most ({place + 1}{_ordinal(place + 1)})"
                if place < 3
                else "something I hold, though not near the top"
            )
            pieces.append(
                Piece(
                    said=said,
                    weight=held or 1.0,
                    source="value",
                    about_her=f"{said} is {standing}",
                )
            )
            offered = int(getattr(value, "offered", 0) or 0)
            chosen = int(getattr(value, "chosen", 0) or 0)
            if offered and int(getattr(value, "lean", 0) or 0) > 0:
                above = float(getattr(value, "rate", 0.0)) - float(
                    getattr(value, "chance", 0.0) or 0.0
                )
                if above > 0.0:
                    pieces.append(
                        Piece(
                            said=said,
                            weight=1.0 + above,
                            source="chose",
                            about_her=(
                                f"when something served {said} I took it "
                                f"{chosen} times out of {offered}, far more "
                                "often than chance would give"
                            ),
                        )
                    )
        for what, times in getattr(portrait, "most_chosen", ()) or ():
            said = str(what or "").replace("_", " ").strip()
            if said:
                pieces.append(
                    Piece(
                        said=said,
                        weight=1.0 + math.log1p(max(0, int(times))) / 10.0,
                        source="chose",
                        about_her=f"what I have done most lately is {said} ({times} times)",
                    )
                )
    try:
        from core.self.stated_preferences import stated_preferences

        for item in stated_preferences(limit=8):
            said = str(getattr(item, "text", "") or "").strip()
            if said:
                pieces.append(
                    Piece(
                        said=said,
                        weight=1.0,
                        source="said",
                        about_her=f'I have said about myself: "{said}"',
                    )
                )
    except Exception as exc:  # noqa: BLE001
        record_degradation("where_i_stand", exc, severity="debug")
    return pieces


def _embedder() -> Any:
    """The organ that turns words into something comparable.

    The registered memory engine holds one and is the shared instance the rest
    of the runtime uses, so the model is loaded once. Where nothing is
    registered — a probe, a test, a process that never booted memory — the
    shared embedding engine is acquired directly rather than doing without,
    because without it there is no measurement at all.
    """
    engine = optional_service("vector_memory_engine", "vector_memory", default=None)
    held = getattr(engine, "embedder", None)
    if hasattr(held, "embed"):
        return held
    if hasattr(engine, "embed"):
        return engine
    try:
        from core.memory.embedding_runtime import acquire_shared_embedding_engine

        shared = acquire_shared_embedding_engine("where-i-stand")
        return shared if hasattr(shared, "embed") else None
    except Exception as exc:  # noqa: BLE001 - no embedder is a measurement of nothing
        record_degradation("where_i_stand", exc, severity="debug")
        return None


def _cosine(left: Any, right: Any) -> float:
    try:
        import numpy as np

        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        if denominator <= 1e-9:
            return 0.0
        return float(np.dot(a, b) / denominator)
    except Exception as exc:  # noqa: BLE001
        logger.debug("the two vectors could not be compared, so their agreement reads 0.0 (%s: %s)",
                     type(exc).__name__, exc)
        return 0.0


def how_much_it_is_her(
    description: str, record: Sequence[Piece] | None = None
) -> tuple[float, tuple[str, ...]]:
    """How much one description matches her record, and what matched.

    A weighted mean of the matches, not the single best one: a side that
    resembles several things she values is more her than one that resembles a
    single thing strongly, and the best-match rule cannot tell those apart.
    """
    said = " ".join(str(description or "").split())
    if not said:
        return 0.0, ()
    pieces = list(record if record is not None else her_record())
    if not pieces:
        return 0.0, ()
    embed = _embedder()
    if embed is None:
        return 0.0, ()
    try:
        asked = embed.embed(said)
    except Exception as exc:  # noqa: BLE001
        record_degradation("where_i_stand", exc, severity="debug")
        return 0.0, ()
    scored: list[tuple[float, Piece]] = []
    for piece in pieces:
        try:
            match = _cosine(asked, embed.embed(piece.said))
        except Exception as exc:  # noqa: BLE001
            record_degradation("where_i_stand", exc, severity="debug")
            continue
        scored.append((match, piece))
    if not scored:
        return 0.0, ()
    total = sum(piece.weight for _match, piece in scored) or 1.0
    mean = sum(match * piece.weight for match, piece in scored) / total
    scored.sort(key=lambda pair: pair[0] * pair[1].weight, reverse=True)
    because = tuple(
        f"{piece.said} ({piece.source}, {match:+.2f})" for match, piece in scored[:3]
    )
    return float(mean), because


def where_she_stands(
    first: str, second: str, record: Sequence[Piece] | None = None
) -> Lean:
    """Which of two descriptions is more her, and by how much.

    Not the gap between two similarity numbers. Everything is somewhat similar
    to everything — measured on her own record, both sides of six real
    dimensions scored between +0.49 and +0.65 — so the gap between two of them
    is a rounding error sitting on a large constant, and a lean built from it is
    always about zero. That is how a measurement lands on the midpoint as surely
    as a guess does.

    What carries the signal is agreement. Each thing her record holds is closer
    to one side or the other, and a lean is how consistently they point the same
    way, weighted by how much each counts. Every piece agreeing is ±1 whatever
    the size of each difference; pieces that cancel are 0. Nothing here needs a
    constant chosen by hand, and the number means something anyone can check:
    the share of her record that leans this way rather than that.
    """
    pieces = list(record if record is not None else her_record())
    embed = _embedder()
    if not pieces or embed is None:
        return Lean(toward=0.0, first=0.0, second=0.0, because=(), measured=False)
    left_said = " ".join(str(first or "").split())
    right_said = " ".join(str(second or "").split())
    if not left_said or not right_said:
        return Lean(toward=0.0, first=0.0, second=0.0, because=(), measured=False)
    try:
        left_vector = embed.embed(left_said)
        right_vector = embed.embed(right_said)
    except Exception as exc:  # noqa: BLE001
        record_degradation("where_i_stand", exc, severity="debug")
        return Lean(toward=0.0, first=0.0, second=0.0, because=(), measured=False)

    leaning: list[tuple[float, float, Piece]] = []
    left_total = 0.0
    right_total = 0.0
    weight_total = 0.0
    for piece in pieces:
        try:
            mine = embed.embed(piece.said)
        except Exception as exc:  # noqa: BLE001
            record_degradation("where_i_stand", exc, severity="debug")
            continue
        to_left = _cosine(mine, left_vector)
        to_right = _cosine(mine, right_vector)
        weight = max(0.0, float(piece.weight))
        leaning.append((to_right - to_left, weight, piece))
        left_total += to_left * weight
        right_total += to_right * weight
        weight_total += weight
    if not leaning or weight_total <= 0.0:
        return Lean(toward=0.0, first=0.0, second=0.0, because=(), measured=False)

    agreement = sum(gap * weight for gap, weight, _piece in leaning)
    disagreement = sum(abs(gap) * weight for gap, weight, _piece in leaning)
    toward = agreement / disagreement if disagreement > 1e-12 else 0.0

    # What she is handed is the things themselves, not the arithmetic over them.
    # Nobody explains a personality answer by quoting a coefficient about
    # themselves: they talk about what they value, what they keep doing, and
    # what they have said before. Only the pieces that lean the way she landed
    # are named, most telling first, because the others are not her reason.
    leaning.sort(key=lambda row: abs(row[0]) * row[1], reverse=True)
    side = 1.0 if toward > 0 else -1.0
    because = tuple(
        piece.about_her or piece.said
        for gap, _weight, piece in leaning
        if gap * side > 0.0
    )[:4]
    return Lean(
        toward=float(max(-1.0, min(1.0, toward))),
        first=float(left_total / weight_total),
        second=float(right_total / weight_total),
        because=because,
        measured=True,
        gap=float(agreement / weight_total),
    )


def themes_among(dimensions: Sequence[str]) -> list[list[int]]:
    """Group dimensions by what they are about, and say which go together.

    A scale instrument asks many questions about a few things. Thinking about
    each of thirty-two items on its own costs thirty-two passes of her
    reasoning and gets thirty-two disconnected sentences; thinking about a
    THEME gets the connected account she gives when someone asks about her in
    conversation, and costs a handful of passes.

    Semantic, and needs nothing about the instrument: a dimension is the words
    of both its ends, and the ones most alike travel together. How many groups
    is the balance between how much she thinks at once and how many times she
    has to think: the square root of the number of items, which is where those
    two costs meet. Chaining them by similarity alone collapses a whole
    instrument into one group — measured on fourteen real dimensions, thirteen
    of them ended up in a single blob — so the size is bounded as well.

    Returns lists of indices into ``dimensions``, in order.
    """
    said = [" ".join(str(one or "").split()) for one in dimensions]
    if len(said) < 3:
        return [[index] for index in range(len(said))]
    embed = _embedder()
    if embed is None:
        return [[index] for index in range(len(said))]
    try:
        vectors = [embed.embed(one) for one in said]
    except Exception as exc:  # noqa: BLE001
        record_degradation("where_i_stand", exc, severity="debug")
        return [[index] for index in range(len(said))]

    wanted = max(1, int(math.ceil(math.sqrt(len(said)))))
    most = max(1, int(math.ceil(len(said) / wanted)))
    groups: list[list[int]] = []
    for index in range(len(said)):
        best: int | None = None
        best_match = -2.0
        for place, members in enumerate(groups):
            if len(members) >= most:
                continue
            match = max(_cosine(vectors[index], vectors[member]) for member in members)
            if match > best_match:
                best_match, best = match, place
        if best is None or len(groups) < wanted and best_match < _typical_match(vectors):
            groups.append([index])
        else:
            groups[best].append(index)
    return groups


def _typical_match(vectors: Sequence[Any]) -> float:
    """The middling similarity among these, so "alike" means alike for this set."""
    matches: list[float] = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            matches.append(_cosine(vectors[left], vectors[right]))
    if not matches:
        return 0.0
    matches.sort()
    return matches[len(matches) // 2]


def which_is_most_her(
    descriptions: Sequence[str], record: Sequence[Piece] | None = None
) -> Choice:
    """Which of several descriptions her own record supports, and by how much.

    The same principle as a lean, for more than two. Comparing how similar each
    description is to her in absolute terms does not work: everything is
    somewhat similar to everything, and the differences are a rounding error on
    a large constant. What carries the signal is which description each thing
    she holds is CLOSEST to — every value, every choice she made when it cost
    something, every sentence she has said about herself lands on one of them,
    weighted by how much it counts, and the shares are what she is asked to
    stand on.

    Nothing here knows what kind of question this is. It takes descriptions of
    a person and answers which one her record is.
    """
    said = [" ".join(str(one or "").split()) for one in descriptions]
    pieces = list(record if record is not None else her_record())
    embed = _embedder()
    if len(said) < 2 or not pieces or embed is None or not all(said):
        return Choice(index=0, support=(), because=(), measured=False)
    try:
        offered = [embed.embed(one) for one in said]
    except Exception as exc:  # noqa: BLE001
        record_degradation("where_i_stand", exc, severity="debug")
        return Choice(index=0, support=(), because=(), measured=False)

    weight_for = [0.0] * len(said)
    supporters: list[list[tuple[float, Piece]]] = [[] for _ in said]
    total = 0.0
    for piece in pieces:
        try:
            mine = embed.embed(piece.said)
        except Exception as exc:  # noqa: BLE001
            record_degradation("where_i_stand", exc, severity="debug")
            continue
        matches = [_cosine(mine, one) for one in offered]
        best = max(range(len(matches)), key=lambda place: matches[place])
        # How much this piece prefers its best over the runner-up, so a piece
        # that barely chooses does not count as much as one that clearly does.
        rest = sorted(matches, reverse=True)
        margin = max(0.0, rest[0] - rest[1]) if len(rest) > 1 else 0.0
        weight = max(0.0, float(piece.weight)) * margin
        weight_for[best] += weight
        supporters[best].append((margin, piece))
        total += weight
    if total <= 0.0:
        return Choice(index=0, support=(), because=(), measured=False)

    support = tuple(weight / total for weight in weight_for)
    index = max(range(len(support)), key=lambda place: support[place])
    supporters[index].sort(key=lambda row: row[0] * row[1].weight, reverse=True)
    because = tuple(
        piece.about_her or piece.said for _margin, piece in supporters[index]
    )[:4]
    return Choice(index=index, support=support, because=because, measured=True)


def against_the_rest(leans: Sequence[Lean]) -> list[float]:
    """Each lean as a share of the strongest one asked of the same record.

    A lean on its own says how consistently her record points one way, and
    consistency saturates: twenty things all a hair closer to one side reads
    the same as twenty things decisively closer. Measured on a real page of
    sixty questions, thirty-four of them came out at the far end, which is not
    a person answering a questionnaire.

    So a page is its own unit. The questions on it are all asked of the same
    record, so the strongest gap among them is what "as far as she goes" means
    here, and every other question is placed in proportion to it. A page she
    genuinely feels strongly about keeps its strong answers; one she is mild
    about stops reading as though she were at the extremes of everything.

    Returns a value in [-1, 1] for each lean, in order. Unmeasured leans give
    0.0 and the caller should not place them at all.
    """
    gaps = [float(lean.gap) if lean.measured else 0.0 for lean in leans]
    widest = max((abs(gap) for gap in gaps), default=0.0)
    if widest <= 1e-12:
        return [0.0 for _ in gaps]
    return [max(-1.0, min(1.0, gap / widest)) for gap in gaps]
