"""What an outcome does to her chemistry.

Lifted whole out of `neurochemical_system`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations


class _AnswersOutcomes:
    """Lifted whole out of NeurochemicalSystem; see neurochemical_system.py."""

    def on_reward(self, magnitude: float=0.3) -> None:
        """Reward received — dopamine + endorphin surge."""
        magnitude = self._event_amount(magnitude, event="reward", default=0.3)
        self.chemicals["dopamine"].surge(magnitude * 0.6)
        self.chemicals["endorphin"].surge(magnitude * 0.3)
        self.chemicals["serotonin"].surge(magnitude * 0.1)

    def on_disappointment(self, magnitude: float=0.3) -> None:
        """Worse than she expected: a dopamine dip, the reward error's other sign.

        A reward prediction error is signed. Dopamine neurons fire above their
        baseline for better than expected and pause below it for worse
        (Schultz, Dayan and Montague 1997), and `on_reward` had only the first
        half. The dip mirrors `on_reward`'s dopamine term and is capped per call
        as `on_threat` caps its depletions. The pause has less room than the
        burst: a neuron firing a few spikes a second can only fall to zero
        (Bayer and Glimcher 2005). Fed by core/affect/what_it_was_worth.py.
        """
        magnitude = self._event_amount(magnitude, event="disappointment", default=0.3)
        self.chemicals["dopamine"].deplete(min(0.08, magnitude * 0.6))

    def on_prediction_error(self, error: float) -> None:
        """Prediction was wrong — norepinephrine + dopamine (learning signal)."""
        error = self._event_amount(error, event="prediction_error", default=0.0)
        self.chemicals["norepinephrine"].surge(error * 0.4)
        self.chemicals["dopamine"].surge(error * 0.3)
        self.chemicals["acetylcholine"].surge(error * 0.2)

    def on_success(self) -> None:
        """Task completed successfully -- also relieves boredom."""
        from .neurochemical_system import (
            _record_neurochemical_degradation,
            logger,
        )

        self.chemicals["dopamine"].surge(0.3)
        self.chemicals["serotonin"].surge(0.15)
        self.chemicals["endorphin"].surge(0.1)
        # Success relieves boredom
        try:
            from core.container import ServiceContainer
            drive = ServiceContainer.get("drive_engine", default=None)
            if drive and hasattr(drive, "relieve_boredom"):
                drive.relieve_boredom("tool_success")
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_neurochemical_degradation(
                exc,
                action="kept success chemistry after drive boredom relief failed",
                severity="warning",
            )
            logger.debug("Drive boredom relief after success failed: %s", exc)

    def on_frustration(self, amount: float=0.3) -> None:
        """Frustration event."""
        amount = self._event_amount(amount, event="frustration", default=0.3)
        self.chemicals["cortisol"].surge(amount * 0.4)
        self.chemicals["norepinephrine"].surge(amount * 0.3)
        self.chemicals["serotonin"].deplete(amount * 0.2)

    def learn_mood_from_outcome(self, observed_mood: dict[str, float]) -> dict[str, float]:
        """Feed an outcome-based mood signal back to the adaptive layer.

        Callers (e.g. action evaluator, homeostasis monitor) pass an empirical
        mood estimate derived from behavior/world-state rather than from the
        chemistry itself, so coefficients can drift away from their seeds.
        """
        from .neurochemical_system import (
            _record_neurochemical_degradation,
            logger,
        )

        self._repair_missing_chemicals()
        chem = {name: float(c.effective) for name, c in self.chemicals.items()}
        try:
            from core.consciousness.adaptive_mood import get_adaptive_mood

            return get_adaptive_mood().update_from_outcome(chem, observed_mood)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            _record_neurochemical_degradation(
                exc,
                action="skipped adaptive mood learning when outcome update failed",
                severity="warning",
            )
            logger.debug("Adaptive mood outcome update failed: %s", exc)
            return {}

