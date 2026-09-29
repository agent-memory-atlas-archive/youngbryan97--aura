"""Two identical samples read as one law, when each anchor's arms are cross-fitted together.

The v25 carrier compares arms forked from the same anchors, so row i of every
arm belongs to anchor i. Cross-fitted without that, a test row's twin from the
other arm sat in the training fold as its nearest neighbour with the other
label, and the classifier read separation between samples that were the same:
a distance squared of 0.056 at 128 anchors and 0.103 at 64. That was the sham
floor of every v25 look, 0.062 to 0.107, and it sat over every singleton cut's
effect. An odd neighbour count then split one twin pair at every point and
left 4/k^2. Grouped folds and an even k take both out.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.intrinsic_v25 import crossfit_fisher_rao
from core.subject.v25_cut import decide_cut, playback_decided

pytestmark = pytest.mark.unit


def _paired(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 40)), rng.normal(size=(n, 30)), np.arange(n)


@pytest.mark.parametrize("n", [16, 64, 128])
def test_identical_samples_read_zero(n: int) -> None:
    context, future, anchors = _paired(n)
    a = np.hstack([context, future])
    assert crossfit_fisher_rao(a, a.copy(), groups=anchors).distance_sq == 0.0


def test_without_the_anchors_the_same_samples_read_as_apart() -> None:
    """The fault this exists for, kept visible: the old folds separate a sample from itself."""
    context, future, _ = _paired(128)
    a = np.hstack([context, future])
    assert crossfit_fisher_rao(a, a.copy()).distance_sq > 0.02


def test_a_real_difference_is_still_seen() -> None:
    context, future, anchors = _paired(128)
    moved = future.copy()
    moved[:, :6] += 1.0
    a = np.hstack([context, future])
    b = np.hstack([context, moved])
    assert crossfit_fisher_rao(a, b, groups=anchors).distance_sq > 0.1


def _samples(n: int, cut_shift: float) -> dict[str, np.ndarray]:
    context, future, _ = _paired(n, seed=3)
    cut = future.copy()
    cut[:, :6] += cut_shift
    return {"context": context, "intact": future, "cut": cut, "sham_a": future, "sham_b": future.copy()}


def test_a_cut_that_changed_nothing_is_not_decided() -> None:
    _estimate, excess, lower, _p = decide_cut(_samples(64, 0.0), tau_seconds=1.0, draws=50, alpha=0.05 / 6, paired=True)
    assert excess == 0.0
    assert lower <= 0.0


def test_a_cut_that_changed_something_is_decided_at_thirty_two_anchors() -> None:
    _estimate, excess, lower, _p = decide_cut(_samples(32, 1.0), tau_seconds=1.0, draws=200, alpha=0.05 / 6, paired=True)
    assert excess > 0.0
    assert lower > 0.0


def test_the_playback_control_is_not_decided() -> None:
    assert not playback_decided(_samples(32, 1.0), tau_seconds=1.0, seed=1, alpha=0.05 / 6, draws=50, paired=True)
