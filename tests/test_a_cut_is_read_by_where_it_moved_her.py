"""A cut is read by how far it moved her from where the same moment went untouched.

`core.subject.paired_displacement` answers the v5 question the way the
preregistration asks it, anchor by anchor, in the units of her own untouched
variation. What it has to get right before any run reads it: identical arms read
nothing, a real move is decided, a cut arm that is only another untouched fork
is not, and neither an orthogonal recoding of the state nor writing every column
twice changes the number.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.paired_displacement import decide, per_anchor
from core.subject.v25_cut import decide_cut, playback_decided

pytestmark = pytest.mark.unit


def _arms(n: int = 64, width: int = 40, move: float = 0.0, noise: float = 0.0, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    untouched = rng.normal(size=(n, width))
    cut = untouched.copy()
    cut[:, :5] += move * rng.normal(size=(n, 5))
    sham_b = untouched + noise * rng.normal(size=(n, width))
    if noise:
        cut = cut + noise * rng.normal(size=(n, width))
    return {"intact": untouched, "cut": cut, "sham_a": untouched, "sham_b": sham_b}


def _decide(slot: dict, **kw):
    return decide(slot, tau_seconds=2.0, seed=1, alpha=0.05 / 6, draws=1000, **kw)


def test_identical_arms_read_nothing() -> None:
    _estimate, excess, lower, _p = _decide(_arms())
    assert excess == 0.0 and lower <= 0.0


def test_a_real_move_is_decided_at_thirty_two_anchors() -> None:
    _estimate, excess, lower, p = _decide(_arms(n=32, move=0.3))
    assert excess > 0.0 and lower > 0.0 and p < 0.05


def test_a_cut_arm_that_is_only_another_fork_is_not_decided() -> None:
    """Forks that part by noise alone: the cut is one more of them."""
    decided = [
        _decide(_arms(noise=0.2, seed=seed))[2] > 0.0 for seed in range(20)
    ]
    assert sum(decided) <= 1


@pytest.mark.parametrize("change", ["rotate", "duplicate"])
def test_the_coordinates_the_state_is_written_in_change_nothing(change: str) -> None:
    slot = _arms(move=0.3, noise=0.05)
    width = slot["intact"].shape[1]
    q, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(width, width)))
    recoded = {
        key: (value @ q if change == "rotate" else np.hstack([value, value]))
        for key, value in slot.items()
    }
    moved, parted = per_anchor(slot["intact"], slot["cut"], slot["sham_a"], slot["sham_b"])
    moved2, parted2 = per_anchor(recoded["intact"], recoded["cut"], recoded["sham_a"], recoded["sham_b"])
    assert np.allclose(moved, moved2, rtol=1e-6) and np.allclose(parted, parted2, rtol=1e-6, atol=1e-9)


def test_the_sweep_uses_it_when_the_design_says_so_and_the_playback_is_never_decided() -> None:
    slot = _arms(n=32, move=0.3) | {"context": np.zeros((32, 3))}
    _estimate, _excess, lower, _p = decide_cut(
        slot, tau_seconds=2.0, seed=1, alpha=0.05 / 6, draws=500, paired=True, estimator="displacement"
    )
    assert lower > 0.0
    assert not playback_decided(
        slot, tau_seconds=2.0, seed=1, alpha=0.05 / 6, draws=500, paired=True, estimator="displacement"
    )
