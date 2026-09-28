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

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from core.runtime.errors import record_degradation
from core.runtime.service_access import optional_service

__all__ = [
    "Lean",
    "Piece",
    "her_record",
    "how_much_it_is_her",
    "where_she_stands",
]


@dataclass(frozen=True)
class Piece:
    """One thing her record says about her, and how much it counts."""

    said: str
    weight: float = 1.0
    source: str = ""


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
        for value in getattr(portrait, "values", ()) or ():
            said = str(getattr(value, "value", "") or "").replace("_", " ")
            if not said:
                continue
            held = max(0.0, float(getattr(value, "held", 0.0) or 0.0))
            pieces.append(Piece(said=said, weight=held or 1.0, source="value"))
            offered = int(getattr(value, "offered", 0) or 0)
            if offered and int(getattr(value, "lean", 0) or 0) > 0:
                above = float(getattr(value, "rate", 0.0)) - float(
                    getattr(value, "chance", 0.0) or 0.0
                )
                if above > 0.0:
                    pieces.append(
                        Piece(said=said, weight=1.0 + above, source="chose")
                    )
        for what, times in getattr(portrait, "most_chosen", ()) or ():
            said = str(what or "").replace("_", " ").strip()
            if said:
                pieces.append(
                    Piece(said=said, weight=1.0 + math.log1p(max(0, int(times))) / 10.0, source="chose")
                )
    try:
        from core.self.stated_preferences import stated_preferences

        for item in stated_preferences(limit=8):
            said = str(getattr(item, "text", "") or "").strip()
            if said:
                pieces.append(Piece(said=said, weight=1.0, source="said"))
    except Exception as exc:  # noqa: BLE001
        record_degradation("where_i_stand", exc, severity="debug")
    return pieces


def _embedder() -> Any:
    engine = optional_service("vector_memory_engine", "vector_memory", default=None)
    return engine if hasattr(engine, "embed") else None


def _cosine(left: Any, right: Any) -> float:
    try:
        import numpy as np

        a = np.asarray(left, dtype=float)
        b = np.asarray(right, dtype=float)
        denominator = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        if denominator <= 1e-9:
            return 0.0
        return float(np.dot(a, b) / denominator)
    except Exception:  # noqa: BLE001
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

    The two matches are on the same scale and measured against the same record,
    so their difference is what carries meaning — the absolute height of either
    one says more about the embedding than about her.
    """
    pieces = list(record if record is not None else her_record())
    left, left_because = how_much_it_is_her(first, pieces)
    right, right_because = how_much_it_is_her(second, pieces)
    if not left_because and not right_because:
        return Lean(toward=0.0, first=0.0, second=0.0, because=(), measured=False)
    gap = right - left
    # Squashed so a small real difference is a small lean rather than a
    # confident one, and so nothing can run away with the scale. The constant
    # is the spread of the differences this measure produces, which is the only
    # thing it could honestly be.
    spread = max(1e-6, (abs(left) + abs(right)) / 2.0)
    toward = math.tanh(gap / spread)
    because = tuple(
        list(left_because[:2]) + list(right_because[:2])
    )
    return Lean(
        toward=float(toward),
        first=float(left),
        second=float(right),
        because=because,
        measured=True,
    )
