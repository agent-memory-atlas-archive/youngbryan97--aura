"""A signal counts once in the estimator, however many columns carry it.

The carrier's duplication control copies the future block and asks whether the
rate rises. On 28 September it rose from 0.0032 to 0.0083 at the weakest cut,
because a copied column doubled its say in the neighbour distance. Counting
each signal once, over both arms and every row, takes that out without looking
at which arm a row is in.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.intrinsic_v25 import crossfit_fisher_rao, intrinsic_rate_from_samples, one_column_per_signal

pytestmark = pytest.mark.unit


def _arms(n: int = 48, seed: int = 2):
    rng = np.random.default_rng(seed)
    intact = rng.normal(size=(n, 10))
    cut = intact.copy()
    cut[:, :2] += 0.7
    return rng.normal(size=(n, 6)), intact, cut


def test_copies_scaled_and_negated_copies_are_one_signal() -> None:
    x = np.random.default_rng(0).normal(size=(20, 3))
    wide = np.hstack([x, x[:, :1] * 3.0 + 2.0, -x[:, 1:2], np.ones((20, 1)), np.zeros((20, 1))])
    kept = one_column_per_signal(wide)
    assert len(kept) == 4  # three signals and one constant


def test_duplicating_the_future_does_not_move_the_rate() -> None:
    context, intact, cut = _arms()
    anchors = np.arange(len(intact))
    kwargs = dict(tau_seconds=2.0, context=context, groups=anchors, seed=3)
    once = intrinsic_rate_from_samples(intact, cut, intact, intact.copy(), one_signal_once=True, **kwargs)
    twice = intrinsic_rate_from_samples(
        np.hstack([intact, intact]), np.hstack([cut, cut]),
        np.hstack([intact, intact]), np.hstack([intact, intact]), one_signal_once=True, **kwargs,
    )
    assert twice.excess_rate == once.excess_rate


def test_without_it_the_copy_is_counted_twice() -> None:
    """The defect, kept visible, so this test fails if the default changes silently."""
    context, intact, cut = _arms()
    anchors = np.arange(len(intact))
    kwargs = dict(tau_seconds=2.0, context=context, groups=anchors, seed=3, one_signal_once=False)
    once = intrinsic_rate_from_samples(intact, cut, intact, intact.copy(), **kwargs)
    twice = intrinsic_rate_from_samples(
        np.hstack([intact, intact]), np.hstack([cut, cut]),
        np.hstack([intact, intact]), np.hstack([intact, intact]), **kwargs,
    )
    assert twice.excess_rate != once.excess_rate


def test_identical_samples_still_read_zero_and_a_difference_is_still_seen() -> None:
    _context, intact, cut = _arms(128)
    anchors = np.arange(len(intact))
    assert crossfit_fisher_rao(intact, intact.copy(), groups=anchors, one_signal_once=True).distance_sq == 0.0
    assert crossfit_fisher_rao(intact, cut, groups=anchors, one_signal_once=True).distance_sq > 0.1
