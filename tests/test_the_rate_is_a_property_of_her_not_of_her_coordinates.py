"""The intrinsic rate must not move when the same state is written differently.

`crossfit_fisher_rao` counts neighbours in Euclidean distance after standardising
each column, and that metric weights the coordinates the state happens to be
written in. On the v25 run of 26 September the rate read 0.019956 raw and
0.010103 under an invertible rotation of the same information, a drift of 0.49,
and the carrier of J* was reported unresolved for it
(docs/WHAT_J_STAR_NEEDS.md).

Whitening makes Euclidean distance equal to Mahalanobis distance, which is the
same number under every invertible linear recoding. These tests hold both halves
of that: that it is invariant, and that it still reads nothing on a sham.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.intrinsic_v25 import crossfit_fisher_rao


def _rotation(width: int, seed: int) -> np.ndarray:
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(width, width)))
    return q


def _two_laws(rows: int, width: int, seed: int, shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Two samples that differ by a shift along one coordinate, on wildly uneven scales.

    The scales matter: her columns run from a variance of 0.001 to 10, and a
    rotation of a block like that is what the invariance check applies.
    """
    rng = np.random.default_rng(seed)
    scale = np.geomspace(0.03, 3.0, width)
    a = rng.normal(size=(rows, width)) * scale
    b = rng.normal(size=(rows, width)) * scale
    b[:, width // 2] += shift * scale[width // 2]
    return a, b


@pytest.mark.parametrize("whiten", [False, True])
def test_a_sham_reads_nothing_either_way(whiten: bool) -> None:
    a, b = _two_laws(400, 6, seed=3, shift=0.0)
    estimate = crossfit_fisher_rao(a, b, seed=1, whiten=whiten)
    assert estimate.distance_sq < 0.25, estimate.distance_sq


@pytest.mark.parametrize("whiten", [False, True])
def test_two_laws_that_differ_are_seen_either_way(whiten: bool) -> None:
    a, b = _two_laws(400, 6, seed=3, shift=3.0)
    apart = crossfit_fisher_rao(a, b, seed=1, whiten=whiten).distance_sq
    same = crossfit_fisher_rao(*_two_laws(400, 6, seed=3, shift=0.0), seed=1, whiten=whiten).distance_sq
    assert apart > same + 0.3, (apart, same)


def test_standardising_each_column_moves_under_a_rotation() -> None:
    """The reading that left the carrier unresolved, on a fixture of the same shape."""
    a, b = _two_laws(400, 6, seed=3, shift=3.0)
    q = _rotation(6, seed=11)
    raw = crossfit_fisher_rao(a, b, seed=1).distance_sq
    turned = crossfit_fisher_rao(a @ q, b @ q, seed=1).distance_sq
    assert abs(raw - turned) / max(raw, 1e-9) > 0.05, (raw, turned)


def test_whitening_is_the_same_number_in_any_basis() -> None:
    a, b = _two_laws(400, 6, seed=3, shift=3.0)
    raw = crossfit_fisher_rao(a, b, seed=1, whiten=True).distance_sq
    for turn in (11, 23, 41):
        q = _rotation(6, seed=turn)
        turned = crossfit_fisher_rao(a @ q, b @ q, seed=1, whiten=True).distance_sq
        assert abs(raw - turned) / max(raw, 1e-9) < 0.05, (raw, turned, turn)


def test_whitening_survives_a_stretch_as_well_as_a_turn() -> None:
    """Mahalanobis distance is invariant to every invertible linear map, not only rotations."""
    a, b = _two_laws(400, 6, seed=3, shift=3.0)
    rng = np.random.default_rng(5)
    mixing = rng.normal(size=(6, 6))
    assert abs(np.linalg.det(mixing)) > 1e-6
    raw = crossfit_fisher_rao(a, b, seed=1, whiten=True).distance_sq
    mixed = crossfit_fisher_rao(a @ mixing, b @ mixing, seed=1, whiten=True).distance_sq
    assert abs(raw - mixed) / max(raw, 1e-9) < 0.05, (raw, mixed)


def test_it_is_not_the_default() -> None:
    """Changing an estimator changes what every earlier run measured."""
    import inspect

    signature = inspect.signature(crossfit_fisher_rao)
    assert signature.parameters["whiten"].default is False
