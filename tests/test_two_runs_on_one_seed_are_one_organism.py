"""`tools/same_organism.py` tells two recordings of one organism from two organisms.

The carrier run of 28 September merged four shards whose recordings differed
from the first frame in 170 to 206 of 438 columns. The tool is how a sharded
design is shown to be one organism before it runs, so it has to see a single
column part at a single frame, and it has to call two identical recordings the
same.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from same_organism import compare  # noqa: E402

pytestmark = pytest.mark.unit


def _run(root: Path, name: str, x: np.ndarray) -> Path:
    run = root / name / "run_001"
    run.mkdir(parents=True)
    np.savez_compressed(run / "core_state.npz", x=x)
    (run / "core_state_manifest.json").write_text(json.dumps({"columns": [f"c{i}" for i in range(x.shape[1])]}))
    return root / name


def test_two_identical_recordings_are_one_organism(tmp_path: Path) -> None:
    x = np.random.default_rng(0).normal(size=(40, 6))
    assert compare(_run(tmp_path, "a", x), _run(tmp_path, "b", x.copy()))["same"] is True


def test_one_column_apart_at_one_frame_is_two(tmp_path: Path) -> None:
    x = np.random.default_rng(0).normal(size=(40, 6))
    y = x.copy()
    y[17, 4] += 1e-12
    report = compare(_run(tmp_path, "a", x), _run(tmp_path, "b", y))
    assert report["same"] is False
    assert report["first_frame_apart"] == 17 and report["columns"] == ["c4"]
