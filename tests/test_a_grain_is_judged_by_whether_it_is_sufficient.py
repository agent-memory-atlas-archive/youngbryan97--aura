"""A grain is judged by whether it stays sufficient, not by a count its spectrum does not define.

At bb3faa54a her signature spectrum fell smoothly from 367 to 150 over twelve
components and met its null near 24, so the rank moved with the number of
anchors (21 or 22 with a fold held out) and with the estimator (60 by
bi-cross-validation), and the carrier refused authority for it. What a grain
has to be is sufficient: once it is known, history adds nothing to the held-out
future. The v5 rule asks that of the grain refitted with each fold held out, and
it still has to fail when history carries what the grain missed.
"""

from __future__ import annotations

import numpy as np
import pytest

import tools.run_subject_core_v25 as runner

pytestmark = pytest.mark.unit


def _world(missing: bool, n: int = 80, seed: int = 4):
    rng = np.random.default_rng(seed)
    state = rng.normal(size=(n, 3))
    hidden = rng.normal(size=(n, 1))
    train = state @ rng.normal(size=(3, 120)) + 0.05 * rng.normal(size=(n, 120))
    future = state @ rng.normal(size=(3, 60))
    if missing:
        # Something history knows that no training signature carries.
        future = future + 4.0 * hidden @ rng.normal(size=(1, 60))
    heldout = future + 0.05 * rng.normal(size=(n, 60))
    history = np.hstack([state, hidden]) @ rng.normal(size=(4, 40)) + 0.05 * rng.normal(size=(n, 40))
    return train, history, heldout


def test_a_grain_that_holds_the_state_is_sufficient_on_every_fold() -> None:
    train, history, heldout = _world(missing=False)
    verdict = runner._sufficiency_stability(train, history, heldout, seed=1)
    assert verdict["sufficiency_stable"] is True and len(verdict["fold_sufficient"]) == 5


def test_a_grain_that_misses_what_history_carries_is_not() -> None:
    train, history, heldout = _world(missing=True)
    assert runner._sufficiency_stability(train, history, heldout, seed=1)["sufficiency_stable"] is False
