"""A condition that holds for a while is said when it changes, not on every look.

Three loops warned on every pass for as long as their condition lasted: the
memory monitor's "HIGH MEMORY PRESSURE" (574 times on 26 September alone), the
memory guard's "RAM CRITICAL ... Strike N", and the hedonic gradient's
"Sustained distress". Each warning after the first said nothing the first had
not, and together they filled the warning channel the neural feed reads, so a
new fault arrived among hundreds of copies of an old one.

The homeostasis engine already reports its strain as it starts and as it ends
(`_report_strain`, 26 September). This is that rule for any loop: the first
look at warning, a look that has worsened by a step at warning again, every
other look at debug, and the end at info.
"""
from __future__ import annotations

import logging


class StandingCondition:
    """One condition, said when it starts, when it worsens by `step`, and when it ends."""

    def __init__(self, logger: logging.Logger, *, ended: str | None, step: float | None = None) -> None:
        self._logger = logger
        self._ended = ended
        self._step = step
        self._active = False
        self._warned_at: float | None = None

    @property
    def active(self) -> bool:
        return self._active

    def report(self, active: bool, message: str, *args: object, level: float | None = None) -> bool:
        """Log this look at the level its change calls for; True when it was a warning.

        `level` is the reading the condition is measured by, when it has one:
        a look `step` above the reading last warned at is worse, and said again.
        """
        if not active:
            if self._active and self._ended:
                self._logger.info(self._ended)
            self._active = False
            self._warned_at = None
            return False
        worse = (
            self._step is not None
            and level is not None
            and self._warned_at is not None
            and level >= self._warned_at + self._step
        )
        if not self._active or worse:
            self._logger.warning(message, *args)
            self._active = True
            self._warned_at = level
            return True
        self._logger.debug(message, *args)
        return False
