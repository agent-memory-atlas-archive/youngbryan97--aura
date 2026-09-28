"""How long a line she says needs to stay up before the next one replaces it.

A commentary a person asked for is for them to read. A loop that narrates and
acts in the same breath puts the next line up before the last was read, and
what a watcher sees is a blur they can only reconstruct from the log
afterwards — which is the thing narration exists to avoid.

The pace is the reading, not the acting: the time comes from how much there is
to read, at an ordinary adult silent reading rate for prose, with a floor so a
short line is still seen.

It belongs to the loops whose tempo is hers to set — a page waits for her
click, so a watcher may as well be given time to read why it came. It does not
belong to a loop with a clock of its own: a live game moves whether or not
anyone has finished reading, and holding a move for five seconds would be
playing the narration instead of the game. So this is offered here and taken by
the loop that can afford it, rather than imposed by the thing that does the
saying.
"""
from __future__ import annotations

import os
import re

#: Ordinary adult silent reading of prose, in words per minute. The rate is a
#: property of the reader, not of what is being narrated, so it is one number
#: for every loop rather than a knob per caller.
WORDS_PER_MINUTE = 200.0

#: The shortest a line stays up, whatever it says. Asked for directly: "pause
#: to give time to read, ~5 seconds" (2026-09-28). A three-word line takes
#: under a second to read and under a second is not time to look.
AT_LEAST_S = 5.0

#: And the longest, so one unusually long line cannot stall a run.
AT_MOST_S = 30.0

_WORD = re.compile(r"\S+")


def time_to_read(text: str) -> float:
    """Seconds to leave ``text`` up before saying the next thing.

    Zero for nothing to read. ``AURA_NARRATION_PACE`` scales it — 0 turns
    pacing off for a run nobody is watching, which is what a batch of tests is.
    """
    words = len(_WORD.findall(str(text or "")))
    if not words:
        return 0.0
    scale = _pace_scale()
    if scale <= 0.0:
        return 0.0
    reading = words / WORDS_PER_MINUTE * 60.0
    return min(AT_MOST_S, max(AT_LEAST_S, reading)) * scale


def _pace_scale() -> float:
    raw = str(os.getenv("AURA_NARRATION_PACE", "") or "").strip()
    if not raw:
        return 1.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 1.0
