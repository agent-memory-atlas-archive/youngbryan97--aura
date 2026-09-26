"""What a turn was worth to her, against what she had come to expect.

Bryan, 26 September: "we need a reward system. what makes it worth it for a
connection to happen and what are the larger effects of that downstream ...
excitement, intensity, satisfaction, surrealness, peace... A payoff. And
detriments to counteract the payoff so that the payoff is preferable and leaned
towards. and of course neutral state ... the gradients between the two. the
nuance. and what expands beyond that binary." And then: it "would give the
systems incentive to connect as well because they benefit or lose out from those
positive/negative signals".

Her chemistry had a reward input and nothing called it; neither was anything
she did ever better or worse than she expected in any way that reached her
connections. They formed from what fired together and from surprise, whether or
not it went anywhere good.

Each turn is read at its end, against the end of the one before, on channels
that are hers already:

    satisfaction   her needs' deficit falling, a need met; rising, one going
                   unmet. Homeostatic reward, the reduction of a drive's
                   distance from where it rests (Keramati and Gutkin 2014)
    accomplishment goals she finished this turn, less goals that failed
    warmth         what being met returned to her drives this turn
    excitement     pleasant and activated, max(0, valence) * arousal
    peace          pleasant and settled, max(0, valence) * (1 - arousal)
    ease           unpleasant and activated, max(0, -valence) * arousal,
                   falling; rising is distress
    spirit         unpleasant and flat, max(0, -valence) * (1 - arousal),
                   falling; rising is gloom
    wonder         her own self-model surprised this turn, signed by how she
                   felt: awe when it was good, something unsettling when it
                   was not

The four affect channels are the quadrants of valence by arousal, the plane
most accounts of affect share (Russell 1980); pleasant and unpleasant
activation move apart rather than as one axis (Watson and Tellegen 1985), so
each quadrant is its own channel. Arousal is the intensity inside them.

A channel is read in units of its own spread, the median size of its changes
over the last turns she has had, so no channel outweighs another by the units
it happens to be kept in. A payoff that is better than she has come to expect
on a channel is a positive error, worse is negative, and as expected is none:
that is the neutral state, and a payoff repeated turn after turn stops being
news, as a reward prediction error does (Schultz, Dayan and Montague 1997). The
expectation is a running mean of what the channel has paid, over the same
window. Each channel keeps its own expectation and error, so what she gets is a
profile rather than one number (Dabney et al. 2020); `worth` is their sum, the
signal the rest of her is taught by.

`worth` is broadcast: a dopamine burst when positive and a dip when negative
(core/consciousness/neurochemical_system.py), dosed by where its size sits in
her own recent history; and it is the third factor her plastic connections
learn by, so a connection that was active before a payoff strengthens and one
active before a detriment weakens (Fremaux and Gerstner 2016). The unified
field's connections are taught this way, the input weights from each organ that
feeds it among them, and so are the substrate's spike-timing traces. An organ
whose activity came before payoffs takes a larger share of the field's input
from the organs whose activity did not, which is what it gains by connecting;
and a source that won her attention during good turns bids stronger for it
(core/affect/what_winning_earned.py).
"""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from core.self.what_came_before import keep_across_stages

__all__ = [
    "CHANNELS",
    "Worth",
    "WorthLedger",
    "broadcast",
    "disabled",
    "dose",
    "get_worth_ledger",
    "read_turn",
    "reset_for_test",
    "teach_connections",
]

CHANNELS: tuple[str, ...] = (
    "satisfaction",
    "accomplishment",
    "warmth",
    "excitement",
    "peace",
    "ease",
    "spirit",
    "wonder",
)

#: How many of her turns a channel's spread, its expectation and the size of
#: her errors are taken over. The same window her other ledgers keep.
_WINDOW = 256

#: Changes a channel needs before its spread means anything.
_ENOUGH = 3


@dataclass
class Worth:
    """One turn's payoffs, what she expected of each, and the difference."""

    payoff: dict[str, float] = field(default_factory=dict)
    expected: dict[str, float] = field(default_factory=dict)
    error: dict[str, float] = field(default_factory=dict)
    worth: float = 0.0
    #: Where the size of this turn's worth sits in her recent history, in [0, 1].
    size: float = 0.0
    turns: int = 0
    measured: bool = False
    why: str = "no turn has been read yet"

    def as_dict(self) -> dict[str, Any]:
        return {
            "payoff": {key: round(value, 6) for key, value in self.payoff.items()},
            "expected": {key: round(value, 6) for key, value in self.expected.items()},
            "error": {key: round(value, 6) for key, value in self.error.items()},
            "worth": round(self.worth, 6),
            "size": round(self.size, 6),
            "turns": self.turns,
            "measured": self.measured,
            "why": self.why,
        }


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return 0.0
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


class WorthLedger:
    """Her channels' recent changes, what each has come to pay, and the last turn's reading."""

    def __init__(self) -> None:
        self._changes: dict[str, deque[float]] = {name: deque(maxlen=_WINDOW) for name in CHANNELS}
        self._expected: dict[str, float] = {name: 0.0 for name in CHANNELS}
        self._paid: dict[str, int] = {name: 0 for name in CHANNELS}
        self._sizes: deque[float] = deque(maxlen=_WINDOW)
        self._last_levels: dict[str, float] | None = None
        #: Goal outcomes already counted, so a goal left finished in the list
        #: is paid for once. The recent ones only: an outcome older than the
        #: window has long left the goal list.
        self._outcomes_seen: deque[str] = deque(maxlen=_WINDOW)
        self._last = Worth()
        self.turns = 0

    def note(self, changes: dict[str, float]) -> Worth:
        """Read one turn's raw changes, one per channel, and return what it was worth."""
        payoff: dict[str, float] = {}
        expected: dict[str, float] = {}
        error: dict[str, float] = {}
        for name in CHANNELS:
            raw = changes.get(name)
            try:
                value = float(raw) if raw is not None else 0.0
            # not a failure: a reading that is not a number is no change on
            # this channel this turn, the neutral reading.
            except (TypeError, ValueError):
                value = 0.0
            if value != value:
                value = 0.0
            history = self._changes[name]
            spread = _median([abs(item) for item in history if item != 0.0])
            if value != 0.0:
                history.append(value)
            if len(history) < _ENOUGH or spread <= 0.0:
                continue
            paid = value / spread
            before = self._expected[name]
            self._paid[name] += 1
            rate = 1.0 / min(self._paid[name], _WINDOW)
            self._expected[name] = before + (paid - before) * rate
            payoff[name] = paid
            expected[name] = before
            error[name] = paid - before
        self.turns += 1
        if not error:
            self._last = Worth(turns=self.turns, why="no channel has changed enough times to have a spread")
            return self._last
        worth = sum(error.values())
        magnitude = abs(worth)
        sizes = list(self._sizes)
        self._sizes.append(magnitude)
        if len(sizes) < _ENOUGH:
            size = 0.0
        else:
            below = sum(1 for item in sizes if item < magnitude)
            ties = sum(1 for item in sizes if item == magnitude)
            size = (below + 0.5 * ties) / len(sizes)
        self._last = Worth(
            payoff=payoff,
            expected=expected,
            error=error,
            worth=worth,
            size=size,
            turns=self.turns,
            measured=True,
            why=(
                f"{'better' if worth > 0 else 'worse' if worth < 0 else 'no different'} than she had "
                f"come to expect across {len(error)} channels, larger than {size:.0%} of her recent turns"
            ),
        )
        return self._last

    def read(self) -> Worth:
        return self._last

    def worth(self) -> float:
        """The signed error her connections learn by, zero until it is measured."""
        return self._last.worth if self._last.measured else 0.0

    def levels_changed(self, levels: dict[str, float]) -> dict[str, float]:
        """This turn's changes of the channels read as levels, against the last turn's end."""
        previous, self._last_levels = self._last_levels, dict(levels)
        if previous is None:
            return {}
        return {name: float(levels[name]) - float(previous[name]) for name in levels if name in previous}


def _goal_outcomes(state: Any, seen: deque[str]) -> float:
    """Goals finished this turn less goals that failed, each counted once."""
    cognition = getattr(state, "cognition", None)
    total = 0.0
    for item in list(getattr(cognition, "active_goals", []) or []):
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "") or "")
        if status not in {"done", "failed"}:
            continue
        key = f"{item.get('id', '') or item.get('goal', '')}:{status}"
        if key in seen:
            continue
        seen.append(key)
        total += 1.0 if status == "done" else -1.0
    return total


def _levels(state: Any) -> dict[str, float]:
    """The channels that are read as the change of a level."""
    levels: dict[str, float] = {}
    budgets = getattr(getattr(state, "motivation", None), "budgets", {}) or {}
    deficits = []
    for entry in budgets.values():
        if not isinstance(entry, dict):
            continue
        capacity = float(entry.get("capacity", 100.0) or 100.0)
        level = float(entry.get("level", capacity) or 0.0)
        if capacity > 0.0:
            deficits.append(max(0.0, capacity - level) / capacity)
    if deficits:
        # A deficit falling is the payoff, so the level is its negative.
        levels["satisfaction"] = -sum(deficits) / len(deficits)
    affect = getattr(state, "affect", None)
    try:
        valence = float(getattr(affect, "valence", 0.0) or 0.0)
        arousal = float(getattr(affect, "arousal", 0.0) or 0.0)
    # not a failure: an affect that is not numbers leaves both channels unread.
    except (TypeError, ValueError):
        return levels
    if not (math.isfinite(valence) and math.isfinite(arousal)):
        return levels
    pleasant, unpleasant = max(0.0, valence), max(0.0, -valence)
    arousal = max(0.0, min(1.0, arousal))
    levels["excitement"] = pleasant * arousal
    levels["peace"] = pleasant * (1.0 - arousal)
    # The unpleasant quadrants pay when they fall, so their levels are negative.
    levels["ease"] = -unpleasant * arousal
    levels["spirit"] = -unpleasant * (1.0 - arousal)
    return levels


def _surprises() -> float:
    """How many times her self-model has been surprised, or nothing when it cannot say."""
    try:
        from core.container import ServiceContainer

        model = ServiceContainer.get("self_prediction", default=None)
        if model is None:
            return 0.0
        return float(model.get_snapshot().get("surprise_count", 0) or 0)
    # not a failure: no self-model is no surprise.
    except (ImportError, AttributeError, TypeError, ValueError):
        return 0.0


def read_turn(state: Any, ledger: WorthLedger | None = None) -> Worth:
    """Read what the turn that just ended was worth to her, and remember it."""
    ledger = ledger or _LEDGER
    levels = _levels(state)
    levels["wonder_count"] = _surprises()
    changes = ledger.levels_changed(levels)
    surprised = changes.pop("wonder_count", 0.0) > 0.0
    try:
        valence = float(getattr(getattr(state, "affect", None), "valence", 0.0) or 0.0)
    # not a failure: a valence that is not a number signs the wonder as good.
    except (TypeError, ValueError):
        valence = 0.0
    if not math.isfinite(valence):
        valence = 0.0
    changes["wonder"] = (1.0 if valence >= 0.0 else -1.0) if surprised else 0.0
    changes["accomplishment"] = _goal_outcomes(state, ledger._outcomes_seen)
    forces = getattr(getattr(state, "motivation", None), "forces", {}) or {}
    try:
        changes["warmth"] = float(forces.get("warmth_return", 0.0) or 0.0)
    except (TypeError, ValueError):
        changes["warmth"] = 0.0
    return ledger.note(changes)


def dose(reading: Worth) -> float:
    """A turn's worth as a signed dose in [-1, 1]: its sign, and its size against her recent turns.

    An ordinary turn is a small dose and her best or worst in a while is a
    large one; a turn no different from what she expected is none.
    """
    if not reading.measured or reading.worth == 0.0:
        return 0.0
    return reading.size if reading.worth > 0.0 else -reading.size


def broadcast(reading: Worth) -> str:
    """Send a measured turn's worth to her chemistry: a burst when better, a dip when worse.

    Returns which way it went, for the caller to record.
    """
    amount = dose(reading)
    if amount == 0.0:
        return "none"
    try:
        from core.container import ServiceContainer

        chemistry = ServiceContainer.get("neurochemical_system", default=None)
    # not a failure: no chemistry registered, as in a bare test process.
    except ImportError:
        return "none"
    if chemistry is None:
        return "none"
    if amount > 0.0:
        chemistry.on_reward(amount)
        return "burst"
    chemistry.on_disappointment(-amount)
    return "dip"


#: Her organs whose connections keep a trace over the turn and are taught by
#: its worth when it ends, by their ServiceContainer names.
_TAUGHT = ("unified_field", "liquid_substrate")


def teach_connections(reading: Worth) -> dict[str, Any]:
    """Let each organ's connections learn what the turn that just ended was worth.

    Every organ is told, a neutral turn included, since the turn's traces end
    with it. The sources that won her attention during the turn are credited
    with it too (core/affect/what_winning_earned.py). Takes the organs' own
    locks, so it runs off the event loop.
    """
    amount = dose(reading)
    try:
        from core.container import ServiceContainer
    # not a failure: no container, as in a bare test process.
    except ImportError:
        return {}
    lessons: dict[str, Any] = {}
    for name in _TAUGHT:
        organ = ServiceContainer.get(name, default=None)
        teach = getattr(organ, "teach", None)
        if callable(teach):
            lessons[name] = teach(amount)
    workspace = ServiceContainer.get("global_workspace", default=None)
    wins = getattr(workspace, "wins_by_source", None)
    if callable(wins):
        from core.affect.what_winning_earned import get_credit_ledger

        lessons["global_workspace"] = get_credit_ledger().note(wins(), amount, measured=reading.measured)
    return lessons


#: The ablation. Set, no turn is read, nothing is sent to her chemistry, and
#: no organ is taught, so the workspace's credit never accrues either. Read on
#: every call rather than at import, so a probe can set it for one process.
DISABLE_ENV = "AURA_DISABLE_PAYOFF"


def disabled() -> bool:
    """Whether the payoff layer is switched off for this process."""
    return os.environ.get(DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def get_worth_ledger() -> WorthLedger:
    return _LEDGER


def reset_for_test() -> None:
    global _LEDGER
    _LEDGER = WorthLedger()


#: Made at import rather than on first use. The subject-core fork carries module
#: globals that hold state, and one still None when an anchor is taken is not
#: carried. See core/soma/good_news.py.
_LEDGER: WorthLedger = WorthLedger()
#: Part of her history, so it is kept across her restarts along her own line.
#: See core/self/what_came_before.py.
keep_across_stages(__name__, "_LEDGER")
