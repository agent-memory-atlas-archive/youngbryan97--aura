"""How the substrate learns.

Lifted whole out of `liquid_substrate`. Every name taken from it is imported at
CALL time: that module imports this one to build the class, and a test that
patches a name on it has to reach the code that reads it.
"""
from __future__ import annotations

import asyncio
from typing import Any


class _HowTheSubstrateLearns:
    """Lifted whole out of LiquidSubstrate; see liquid_substrate.py."""

    async def _apply_plasticity(self) -> None:
        await asyncio.to_thread(self._apply_plasticity_sync)

    def _apply_plasticity_sync(self) -> None:
        """Hebbian learning, and the spike timing the turn's worth will judge.

        Two learning signals are combined:
        1. Base Hebbian (coactivity-driven, always active)
        2. STDP: every step lays down eligibility traces from spike timing,
           and the turn's worth, when it arrives, decides whether they
           strengthen or weaken. See teach().

        The STDP reward used to be delivered here, every step, as the negative
        of the free-energy engine's `prediction_error`. That state has no such
        field, so the reward read zero on every step of every run: no weight
        ever moved, and the zero deltas drove each synapse's uncertainty down
        until the engine had identity-locked them all. The same call held the
        weights inside their spectral and homeostatic bounds, which the
        Hebbian update above needs, so that part still runs every step.
        """
        from .liquid_substrate import (
            logger,
            np,
            record_degradation,
        )

        with self.sync_lock:
            # Numerical Stability: Use tanh on coactivity to prevent runaway growth
            coactivity = np.tanh(np.outer(self.x, self.x))

            # 1. Base Hebbian update
            self.W += self.config.hebbian_rate * coactivity

            # 2. Spike timing into the eligibility traces (from BrainCog research)
            try:
                stdp = self._stdp_engine()
                if stdp is not None:
                    stdp.record_spikes(self.x, t=self.tick_count * 50.0)
                    self.W = stdp.regulate(self.W)
            except (ImportError, AttributeError, RuntimeError) as e:
                record_degradation("liquid_substrate", e)
                logger.debug("STDP plasticity step skipped: %s", e)

            # 3. Neural Resonance: Slow weight calibration towards high-phi states
            if hasattr(self, "_current_phi") and self._current_phi > 0.5:
                resonance_gain = self.config.hebbian_rate * 0.1
                limited_phi = min(10.0, self._current_phi)
                self.W += resonance_gain * coactivity * limited_phi

            # Purge NaN/Inf
            self.W = np.nan_to_num(self.W, nan=0.0, posinf=5.0, neginf=-5.0)

            # Normalization & clipping
            norm = np.linalg.norm(self.W)
            if norm > 10.0:
                self.W *= 10.0 / norm
            self.W = np.clip(self.W, -5.0, 5.0)
            self._cached_connectivity_norm = float(np.linalg.norm(self.W))
            self._mark_weight_cache_dirty()

    def _stdp_engine(self) -> Any:
        """The process's STDP engine, sized to this substrate."""
        from core.consciousness.stdp_learning import get_stdp_engine
        from core.container import ServiceContainer

        from .liquid_substrate import (
            np,
        )

        substrate_neurons = int(np.asarray(self.x).size)
        stdp = (
            ServiceContainer.get("stdp_engine", default=None)
            if ServiceContainer.has("stdp_engine")
            else None
        )
        if stdp is None or int(getattr(stdp, "n", 0) or 0) != substrate_neurons:
            stdp = get_stdp_engine(n_neurons=substrate_neurons)
        return stdp

    def teach(self, modulator: float) -> dict[str, float]:
        """Deliver a turn's worth to the synapses whose spike timing is still eligible.

        `modulator` is the turn's worth as her chemistry got it, signed, in
        [-1, 1] (core/affect/what_it_was_worth.py). Surprise from the
        free-energy engine sets the step size, as it did when the reward was
        delivered every step.
        """
        from .liquid_substrate import (
            np,
            record_degradation,
        )

        with self.sync_lock:
            try:
                stdp = self._stdp_engine()
                if stdp is None:
                    return {"modulator": 0.0, "moved": 0.0}
                surprise = 0.0
                from core.container import ServiceContainer

                engine = ServiceContainer.get("free_energy_engine", default=None)
                current = getattr(engine, "current", None) if engine is not None else None
                if current is not None:
                    surprise = float(getattr(current, "surprise", 0.0) or 0.0)
                dw = stdp.deliver_worth(modulator, surprise)
                before = self.W.copy()
                self.W = stdp.apply_to_connectivity(self.W, dw)
                self._cached_connectivity_norm = float(np.linalg.norm(self.W))
                self._mark_weight_cache_dirty()
                return {
                    "modulator": float(stdp._last_reward),
                    "surprise": surprise,
                    "moved": float(np.abs(self.W - before).sum()),
                }
            except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as e:
                record_degradation("liquid_substrate", e, action="left this turn's worth undelivered to the substrate")
                return {"modulator": 0.0, "moved": 0.0}

