"""The grain's rows, shared out anchor by anchor, are the rows one process makes.

On 28 September the coordinator spent 6 h 20 min learning the grain alone while
four shard workers finished their cuts in 45 minutes and exited. With
`--grain-claims` every process claims anchors one at a time and writes their
signature rows into the shard directory, and the coordinator gathers them. That
is only a faster grain if it is the same grain: a row is used only if it was made
from this anchor's state and these doses, a row that was not is made again, a
claim nobody finished is taken back, and a refused sweep stops the gather.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import tools.run_subject_core_v25 as runner

pytestmark = pytest.mark.unit

DOSES = {"A": 0.5, "P": 1.25}
PLAN = {"train": ("t1", "t2"), "test": ("h1",), "frequencies": np.zeros(1)}


def _anchors(n: int = 6) -> list[SimpleNamespace]:
    rng = np.random.default_rng(3)
    return [SimpleNamespace(current=rng.normal(size=4), snapshot=index) for index in range(n)]


@pytest.fixture
def signature(monkeypatch):
    """A signature that depends only on the anchor and the actions, and counts its calls."""
    calls: list[int] = []

    async def fake(runtime, anchors, conditions, actions, **_kw):
        rows = []
        for anchor in anchors:
            calls.append(int(anchor.snapshot))
            rows.append(np.concatenate([anchor.current * len(actions), [float(len(actions))]]))
        return np.vstack(rows)

    import core.subject.v25_grain as grain

    monkeypatch.setattr(grain, "signature_matrix", fake)
    return calls


def _gather(directory: Path, anchors, refused: Path | None = None):
    return asyncio.run(runner._gather_grain_rows(
        None, anchors, [], PLAN, {}, directory=directory, refused=refused, wait_seconds=60.0, doses=DOSES,
    ))


def _one_process(anchors):
    import core.subject.v25_grain as grain

    train = asyncio.run(grain.signature_matrix(None, anchors, [], PLAN["train"]))
    heldout = asyncio.run(grain.signature_matrix(None, anchors, [], PLAN["test"]))
    return train, heldout


def test_alone_the_coordinator_makes_every_row_as_one_process_would(tmp_path, signature) -> None:
    anchors = _anchors()
    train, heldout, shared = _gather(tmp_path, anchors)
    expected_train, expected_heldout = _one_process(anchors)
    assert np.array_equal(train, expected_train) and np.array_equal(heldout, expected_heldout)
    assert shared == 0


def test_rows_another_process_made_are_used_and_counted(tmp_path, signature) -> None:
    anchors = _anchors()
    other = asyncio.run(runner._work_grain_rows(
        None, anchors[:3], [], PLAN, {}, directory=tmp_path, refused=None, doses=DOSES,
    ))
    assert other == 3
    signature.clear()
    train, heldout, shared = _gather(tmp_path, anchors)
    assert shared == 3 and sorted(set(signature)) == [3, 4, 5]
    expected_train, expected_heldout = _one_process(anchors)
    assert np.array_equal(train, expected_train) and np.array_equal(heldout, expected_heldout)


def test_a_row_made_from_another_state_is_made_again(tmp_path, signature) -> None:
    anchors = _anchors()
    impostor = [SimpleNamespace(current=a.current + 1.0, snapshot=a.snapshot) for a in anchors[:2]]
    asyncio.run(runner._work_grain_rows(None, impostor, [], PLAN, {}, directory=tmp_path, refused=None, doses=DOSES))
    train, _heldout, shared = _gather(tmp_path, anchors)
    assert shared == 0
    assert np.array_equal(train, _one_process(anchors)[0])


def test_a_refused_sweep_stops_the_gather(tmp_path, signature) -> None:
    refused = tmp_path / "REFUSED"
    refused.write_text("PAGCSMWDN|I")
    assert _gather(tmp_path / "grain", _anchors(), refused=refused) is None


def test_a_claim_nobody_finished_is_taken_back(tmp_path, signature, monkeypatch) -> None:
    anchors = _anchors(3)
    assert runner._claim_grain_row(tmp_path, 1)  # a worker that took anchor 1 and died
    # The runner's own clock only: `runner.time` is the time module itself.
    from tests.clock_patch import patch_module_clock

    patch_module_clock(monkeypatch, runner, monotonic=_clock_that_runs_fast())
    train, _heldout, _shared = _gather(tmp_path, anchors)
    assert np.array_equal(train, _one_process(anchors)[0])


def _clock_that_runs_fast():
    now = [0.0]

    def tick() -> float:
        now[0] += 10.0
        return now[0]

    return tick
