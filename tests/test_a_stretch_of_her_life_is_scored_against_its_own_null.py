"""`phi_do`'s folds fit on the past and test on the future, and she learns as a run goes on.

`tools/audit_stationary_irreducibility.py` scores the same cut on each
contiguous stretch instead, and beside each stretch it scores a row-shuffled
copy: the same rows with every relation between them gone. Without that null a
positive within-stretch reading says nothing, because a short window of a
drifting recording can lose prediction to a cut for no reason but the drift.

On `whole-s7-27dc1dda9` the forward-chaining folds were
`[-0.382659, 1e-05, 1e-06, -0.001923, 0.002395]` at `PIM|AGCSWDN`. Each
contiguous fifth at the same cut read +0.003557, +0.005177, +0.002228,
+0.008695 and +0.006964, all with positive lower bounds, and every fifth of the
shuffled copy read at or below zero.

These tests hold the tool's own arithmetic, not that run: a recording built with
a coupling only in its second half, and its null.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.irreducibility import phi_do
from core.subject.recording import Recording
from core.subject.state import DOMAINS, domain_slices
from tools.audit_stationary_irreducibility import rows_of, shuffled, stretches


def _recording(frames: int, *, coupled_from: int, seed: int = 0) -> Recording:
    """P drives A, but only from `coupled_from` on. Every other column is noise.

    The estimator reads a domain through its leading components, so a coupling
    carried by one column out of three hundred is under its own noise: at one
    column a run of six thousand rows read 0.000869 against a shuffled 0.000712.
    Every column of both domains carries the drive here, which is what a domain
    that is genuinely coupled to another looks like to it.
    """
    rng = np.random.default_rng(seed)
    slices = domain_slices()
    width = max(piece.stop for piece in slices.values())
    x = rng.normal(0.0, 1.0, size=(frames, width))
    drive = np.cumsum(rng.normal(0.0, 1.0, size=frames)) * 0.05
    source = slices["P"]
    x[:, source] = drive[:, None] + rng.normal(0.0, 0.2, size=(frames, source.stop - source.start))
    coupled = np.arange(frames) >= coupled_from
    sink = slices["A"]
    x[np.ix_(coupled, np.arange(sink.start, sink.stop))] = drive[coupled, None] * 3.0 + rng.normal(
        0.0, 0.2, size=(int(coupled.sum()), sink.stop - sink.start)
    )
    return Recording(
        x=x,
        conditions=("rest",) * frames,
        tags=("",) * frames,
        times=np.arange(frames, dtype=np.float64),
        env=np.zeros((frames, 1)),
        env_names=("turn",),
        columns=tuple(f"c{index}" for index in range(width)),
        slices=slices,
        notes={},
    )


def test_stretches_are_contiguous_and_cover_the_run() -> None:
    recording = _recording(1_000, coupled_from=500)
    bounds = stretches(recording, 5)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 1_000
    assert [low for _, low in bounds[:-1]] == [high for high, _ in bounds[1:]]


def test_a_stretch_carries_everything_the_rows_carried() -> None:
    """A stretch is the recording restricted to rows, not a bare array."""
    recording = _recording(200, coupled_from=100)
    part = rows_of(recording, np.arange(50, 90))
    assert part.frames == 40
    assert part.width == recording.width
    assert part.columns == recording.columns
    assert part.slices == recording.slices
    assert np.allclose(part.x, recording.x[50:90])
    assert np.allclose(part.times, recording.times[50:90])


def test_shuffling_rows_keeps_every_column_and_loses_every_relation() -> None:
    recording = _recording(400, coupled_from=0)
    fake = shuffled(recording, seed=3)
    assert fake.frames == recording.frames
    for index in range(recording.width):
        assert np.allclose(np.sort(fake.x[:, index]), np.sort(recording.x[:, index]))
    assert not np.allclose(fake.x, recording.x)
    # The clock is not evidence about an order the shuffle destroyed.
    assert np.allclose(fake.times, recording.times)


def test_the_stretch_with_the_coupling_scores_above_its_shuffled_null() -> None:
    """The reading the tool reports: real above zero, shuffled at or below it."""
    recording = _recording(6_000, coupled_from=3_000, seed=11)
    cut = (("P",), tuple(name for name in DOMAINS if name != "P"))
    fake = shuffled(recording, seed=11)
    late = np.arange(3_000, 6_000)
    real = phi_do(rows_of(recording, late), at=cut).as_dict()
    null = phi_do(rows_of(fake, late), at=cut).as_dict()
    assert real["phi_do"] > 0.0, real
    assert null["phi_do"] == pytest.approx(0.0, abs=real["phi_do"] / 2.0), (real, null)
    assert null["lower_bound"] <= real["lower_bound"], (real, null)
