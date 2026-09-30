"""A subject-core run declares its entropy; it does not draw the machine's.

The predictive self model adds managed entropy to every observation before it
updates its weights, and the source was the ANU quantum generator over the
network, then `os.urandom`: unseeded, uncarried, and different in every
process. The run's own generator is seeded and carried by every snapshot.
"""

from __future__ import annotations

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

from core.runtime.managed_entropy import ManagedEntropy, get_managed_entropy
from core.subject.driver import install_declared_host, release_declared_host

pytestmark = pytest.mark.unit


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(soma=SimpleNamespace(hardware={})), declared_host=None)


def test_a_declared_run_draws_its_noise_from_its_own_seed(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a declared run drew from the machine")

    monkeypatch.setattr(os, "urandom", refuse)
    runtime = _runtime()
    install_declared_host(runtime)
    try:
        source = get_managed_entropy()
        random.seed(23)
        first = source.get_prediction_noise(8)
        random.seed(23)
        again = source.get_prediction_noise(8)
        np.testing.assert_array_equal(first, again)
    finally:
        release_declared_host(runtime)


def test_releasing_the_host_gives_the_source_back() -> None:
    runtime = _runtime()
    install_declared_host(runtime)
    source = get_managed_entropy()
    assert "_raw_float" in vars(source)
    release_declared_host(runtime)
    assert "_raw_float" not in vars(source)
    assert source._raw_float.__func__ is ManagedEntropy._raw_float


def test_the_fork_carries_the_budget() -> None:
    from core.subject.snapshot import _singleton_state

    source = get_managed_entropy()
    source._budget_remaining = 0.0123
    carried = _singleton_state()["runtime.managed_entropy"]
    assert carried["_budget_remaining"] == 0.0123
