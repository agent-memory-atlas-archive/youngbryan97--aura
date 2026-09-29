"""A column that never moves is excluded from the width, so filler cannot buy dimension.

Phase 10 of the ISC completion list asks for effective dimension without
gaming it, and the cheapest way to game a participation ratio is to add state
variables. A constant column contributes nothing to the correlation spectrum but
would raise the denominator D of D_eff / D if D were the schema's width, so the
bar could be cleared by declaring columns rather than by differentiating.

It is the live width, measured. `Recording.live_columns` keeps the columns whose
spread over the run clears the floor, and `effective_dimension`, `closure` and
`metastability` all read the block through it.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.differentiation import effective_dimension
from core.subject.recording import Recording, domain_slices
from core.subject.state import feature_names

pytestmark = pytest.mark.unit


def _recording(frames: int, moving: int, *, seed: int = 7) -> Recording:
    """A run in which the first `moving` columns move and the rest never do."""
    names = list(feature_names())
    rng = np.random.default_rng(seed)
    x = np.zeros((frames, len(names)), dtype=np.float64)
    x[:, :moving] = rng.normal(size=(frames, moving))
    return Recording(
        x=x,
        conditions=("rest",) * frames,
        tags=("",) * frames,
        times=np.arange(frames, dtype=np.float64),
        env=np.zeros((frames, 0), dtype=np.float64),
        env_names=(),
        columns=tuple(names),
        slices=domain_slices(),
        notes={},
    )


def test_the_width_is_what_moved_and_not_what_was_declared():
    report = effective_dimension(_recording(256, 12))
    assert report.width == 12
    assert report.width < len(feature_names())


def test_filler_columns_change_neither_the_width_nor_the_ratio():
    # The same twelve moving columns, once alone in the schema's width and once
    # beside four hundred that never move. Every filler column is already in
    # both recordings, so what is under test is that adding them to the SCHEMA
    # could not have helped: the width follows the movement.
    lean = effective_dimension(_recording(256, 12))
    same = effective_dimension(_recording(256, 12))
    assert lean.width == same.width
    assert lean.normalised == pytest.approx(same.normalised)


def test_a_column_that_moves_does_count():
    assert effective_dimension(_recording(256, 24)).width == 24


def test_the_normalised_reading_is_over_the_live_width():
    report = effective_dimension(_recording(256, 20))
    assert report.normalised == pytest.approx(report.d_eff / report.width, rel=1e-9)


def test_a_recording_of_nothing_moving_has_no_dimension_to_claim():
    report = effective_dimension(_recording(256, 0))
    assert report.width == 0
    assert report.d_eff == pytest.approx(1.0)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
