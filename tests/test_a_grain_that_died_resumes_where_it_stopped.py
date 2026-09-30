"""A grain row already made is not made again, so a death inside the grain resumes.

The grain is the longest stage of a v5 run — 11,264 rollouts for the training
matrix and 15,360 for the attack on ten domains, against about 4,100 for the
whole 511-cut sweep — and its checkpoint is written only once it holds every
anchor. A run that died at the last anchor lost all of them.

The claim machinery already writes a row per anchor and reuses only rows made
from the same anchor state and the same doses, which is what a resume needs. It
lived under `--from-shards` because sharding was the first thing to want it.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import tools.run_subject_core_v25 as runner

pytestmark = pytest.mark.unit


def test_a_lone_run_keeps_its_rows_under_the_run_directory():
    source = inspect.getsource(runner.main) if hasattr(runner, "main") else ""
    body = source or Path_source()
    assert "else (run_dir if args.grain_claims else None)" in body


def Path_source() -> str:
    from pathlib import Path

    return Path(inspect.getsourcefile(runner)).read_text()


def test_a_lone_run_waits_for_nobody():
    assert "wait_seconds=0.0 if args.from_shards is None" in Path_source()


def test_a_row_is_evidence_only_for_its_own_anchor_and_doses(tmp_path):
    directory = tmp_path / "grain"
    current = np.array([0.5, 0.25])
    doses = {"P": 1.0, "I": 2.0}
    assert runner._claim_grain_row(directory, 0)
    runner._write_grain_row(
        directory, 0, np.array([1.0, 2.0]), np.array([3.0, 4.0]), current, doses
    )
    row = runner._read_grain_row(directory, 0)
    assert row is not None

    assert runner._row_is_this_anchors(row, current, doses)
    # A different anchor's state, or the same anchor under different doses, is a
    # different measurement wearing the right filename.
    assert not runner._row_is_this_anchors(row, current + 1.0, doses)
    assert not runner._row_is_this_anchors(row, current, {"P": 9.0, "I": 2.0})
    assert not runner._row_is_this_anchors(row, current, {"P": 1.0})
    # And the order doses were written in cannot change the answer.
    assert runner._row_is_this_anchors(row, current, {"I": 2.0, "P": 1.0})


def test_a_row_missing_its_provenance_is_not_evidence():
    assert not runner._row_is_this_anchors({"train": np.array([1.0])}, np.array([0.0]), {})


def test_a_claim_is_taken_once(tmp_path):
    directory = tmp_path / "grain"
    assert runner._claim_grain_row(directory, 3)
    assert not runner._claim_grain_row(directory, 3)


def test_an_unclaimed_anchor_reads_as_absent(tmp_path):
    assert runner._read_grain_row(tmp_path / "grain", 7) is None


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
