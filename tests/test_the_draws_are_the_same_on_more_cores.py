"""Spreading a cut's draws over processes changes the wall clock and nothing else.

`core.subject.draw_pool` evaluates bootstrap and permutation draws across
processes. The claim it makes is that every number is the number one process
computes, in the same order; if a single draw came back different, a lower
bound or a p-value could move across a decision, and a faster sweep would be a
different instrument.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.intrinsic_v25 import bootstrap_rate_difference, paired_permutation_pvalue

pytestmark = pytest.mark.unit


def _arms(n: int = 24, seed: int = 5) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    intact = rng.normal(size=(n, 12))
    cut = intact.copy()
    cut[:, :3] += 0.8
    return rng.normal(size=(n, 10)), intact, cut, intact, intact.copy()


def test_the_bootstrap_is_the_same_on_three_processes(monkeypatch) -> None:
    context, intact, cut, sham_a, sham_b = _arms()
    kwargs = dict(tau_seconds=2.0, context=context, draws=30, seed=11, paired=True)
    monkeypatch.setenv("AURA_ESTIMATOR_WORKERS", "1")
    one = bootstrap_rate_difference(intact, cut, sham_a, sham_b, **kwargs)
    monkeypatch.setenv("AURA_ESTIMATOR_WORKERS", "3")
    three = bootstrap_rate_difference(intact, cut, sham_a, sham_b, **kwargs)
    assert np.array_equal(one, three)


def test_the_permutation_test_is_the_same_on_three_processes(monkeypatch) -> None:
    context, intact, cut, sham_a, sham_b = _arms()
    kwargs = dict(tau_seconds=2.0, context=context, draws=29, seed=13, paired=True)
    monkeypatch.setenv("AURA_ESTIMATOR_WORKERS", "1")
    one = paired_permutation_pvalue(intact, cut, sham_a, sham_b, **kwargs)
    monkeypatch.setenv("AURA_ESTIMATOR_WORKERS", "3")
    three = paired_permutation_pvalue(intact, cut, sham_a, sham_b, **kwargs)
    assert one == three
