from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.kernel.bridge import Phase
from core.runtime.errors import record_degradation
from core.state.aura_state import AuraState

if TYPE_CHECKING:
    from core.kernel.aura_kernel import AuraKernel

logger = logging.getLogger("Aura.SelfReview")

class SelfReviewPhase(Phase):
    """
    Evidence-bounded self-review phase.
    Aura analyzes runtime performance, technical debt, and code health.
    """

    def __init__(self, kernel: AuraKernel):
        super().__init__(kernel)
        self._last_review_ts = 0.0
        self._review_interval = 600.0 # 10 minutes
        self._flagged = False

    async def execute(self, state: AuraState, objective: str | None = None, **kwargs) -> AuraState:
        """
        Analyze current state for potential self-optimization.
        Does not block user interaction — performs meta-analysis on the side.
        """
        # Ten minutes of her life, on the clock her other phases read. The
        # event loop's clock is the machine's, and a measurement run keeps its
        # own: restored with everything else at each arm, this read the time
        # the machine had taken since the snapshot and reviewed on the first
        # turn of every arm, 3,363 times in one run of 26 September.
        now = time.time()
        if now - self._last_review_ts < self._review_interval:
            return state

        # [TUNNELING] Analyze code debt & logic bottlenecks
        logger.info("🧠 [SELF-REVIEW] Initiating bounded self-review.")
        
        # Guard against kernel not having loop_state() yet (early boot calls)
        loop_state_fn = getattr(self.kernel, "loop_state", None)
        if not callable(loop_state_fn):
            logger.debug("SelfReview: kernel.loop_state() not available yet. Skipping.")
            return state

        try:
            loop_state = loop_state_fn()
        except (RuntimeError, AttributeError, TypeError, ValueError) as e:
            record_degradation('self_review', e)
            logger.warning("SelfReview: loop_state() raised: %s", e)
            return state

        phi = loop_state.get("phi", 0.0)
        entropy = loop_state.get("entropy", 0.0)

        # The finding is reported when it starts and when it ends. It used to
        # file an "architectural_review" intent as well, for a self-modification
        # engine that has no way to take one: nothing in the tree read that
        # intent type, and the list it went into keeps its last twenty, so
        # every review pushed out an intent something does read, such as the
        # autotelic objectives the research cycle looks for there.
        flagged = entropy > 0.7 or phi < 0.2
        if flagged and not self._flagged:
            logger.warning("📉 [SELF-REVIEW] High entropy or low phi (phi=%.3f, entropy=%.3f).", phi, entropy)
        elif flagged:
            logger.debug("SelfReview: still high entropy or low phi (phi=%.3f, entropy=%.3f).", phi, entropy)
        elif self._flagged:
            logger.info("🧠 [SELF-REVIEW] Phi and entropy are back in range (phi=%.3f, entropy=%.3f).", phi, entropy)
        self._flagged = flagged

        self._last_review_ts = now
        return state
