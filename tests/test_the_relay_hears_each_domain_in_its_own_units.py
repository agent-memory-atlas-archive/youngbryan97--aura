"""The relay hears each domain in its own units, and never hears a clock.

`core/consciousness/domain_relay.py` squashed each domain's raw numbers with a
tanh, and her numbers run from a share to a step count: C through a loop
counter, N through ontogeny's step count, P through an objective's length in
characters. Those domains sat at 1.0 for good, so the relay mostly added a fixed
pattern and a cut that removed a domain removed nothing. `RelayScope` puts each
field in units of its own recent spread and leaves out a field that has only
ever grown.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from core.consciousness.domain_relay import DomainRelay, RelayScope

pytestmark = pytest.mark.unit


def _readings(step: int, valence: float) -> dict[str, list[float]]:
    return {"C": [float(1000 + step), valence], "N": [float(50_000 + 7 * step)], "A": [valence]}


def test_a_count_that_only_grows_never_reaches_the_relay() -> None:
    scope = RelayScope()
    for step in range(40):
        out = scope.standardise(_readings(step, 0.1 * ((-1) ** step)))
    assert out["N"] == []
    assert len(out["C"]) == 1


def test_a_field_is_read_in_units_of_its_own_spread() -> None:
    scope = RelayScope()
    rng = np.random.default_rng(0)
    for step in range(300):
        out = scope.standardise(_readings(step, float(rng.normal(scale=0.01))))
    assert abs(out["A"][0]) < 5.0
    scope.standardise(_readings(300, 0.0))
    far = scope.standardise(_readings(301, 0.05))["A"][0]
    assert far > 3.0


def test_a_domain_contributes_by_how_far_it_has_moved_rather_than_by_its_units() -> None:
    relay = DomainRelay(64)
    quiet = relay.drive({"A": [0.1]})
    loud = relay.drive({"A": [3.0]})
    assert np.linalg.norm(loud) > np.linalg.norm(quiet)


def test_a_fork_carries_what_the_relay_has_seen() -> None:
    scope = RelayScope()
    for step in range(20):
        scope.standardise(_readings(step, 0.1 * ((-1) ** step)))
    forked = copy.deepcopy(scope)
    assert forked.standardise(_readings(20, 0.3)) == scope.standardise(_readings(20, 0.3))
