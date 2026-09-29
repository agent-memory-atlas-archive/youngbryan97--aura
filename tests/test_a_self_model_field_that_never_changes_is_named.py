"""A self-model field that never changes is named, not counted as wired.

Phase 45 of the ISC completion list rechecks every kind of dead mechanism this
project has found. A self-model field with a writer that never bites looks
exactly like one whose value is genuinely constant, and both look like a field
that is working: something reads it, something writes it, and the graph says the
edge is there.

The recording answers it by measurement rather than by inspection. A column
whose spread over the whole run is below the floor did not move, whatever the
code says, and the run names each one instead of reporting a count.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.recording import FLAT_EPS, Recording, domain_slices
from core.subject.state import feature_names

pytestmark = pytest.mark.unit


def _recording(frames: int = 64, moved: tuple[str, ...] = ()) -> Recording:
    """A run in which only the named columns moved."""
    names = list(feature_names())
    x = np.zeros((frames, len(names)), dtype=np.float64)
    rng = np.random.default_rng(7)
    for name in moved:
        x[:, names.index(name)] = rng.normal(size=frames)
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


def _a_self_model_column() -> str:
    return next(name for name in feature_names() if name.startswith("S."))


def test_a_self_model_field_that_never_moved_is_named():
    column = _a_self_model_column()
    assert column in _recording().flat_columns()


def test_one_that_moved_is_not():
    column = _a_self_model_column()
    assert column not in _recording(moved=(column,)).flat_columns()


def test_every_flat_column_is_named_rather_than_counted():
    recording = _recording()
    assert len(recording.flat_columns()) == len(recording.columns)
    assert set(recording.flat_columns()) <= set(recording.columns)


def test_the_run_reports_the_names_beside_the_count():
    summary = _recording().summary()
    assert summary["flat_columns"] == len(summary["flat_column_names"])
    assert _a_self_model_column() in summary["flat_column_names"]


def test_a_field_that_barely_moves_is_still_flat():
    column = _a_self_model_column()
    names = list(feature_names())
    recording = _recording()
    recording.x[:, names.index(column)] = FLAT_EPS / 10.0
    assert column in recording.flat_columns()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
