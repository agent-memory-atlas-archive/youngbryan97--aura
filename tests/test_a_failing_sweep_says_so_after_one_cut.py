"""A sweep whose claim is going to fail says so after one cut, not after all of them.

Irreducibility is a conjunction over every bipartition, so a single cut left
undecided at its last look refuses it. The breadth-first sweep took every cut
through each look together and only knew at the end, which on her 511 cuts was
a machine-week. `fail_fast` takes the most lopsided cuts first, each to its
decision, and stops at the first that ends undecided. Every cut it does score
reads what the full sweep would have read, because a cut's samples and seed
depend on its place in the full list and not on when it was taken.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.subject import v25_cut
from core.subject.v25_cut import SweepReport, merge_sweeps, sweep_cuts_over_lags

DOMAINS = ("A", "B", "C", "D")


def _anchors(count: int) -> list:
    return [SimpleNamespace(snapshot=index, current=np.zeros(2)) for index in range(count)]


@pytest.fixture
def scored(monkeypatch) -> dict[str, list[int]]:
    """Fake rollouts and a fake decision: each cut's lower bound is set by name."""
    taken: dict[str, list[int]] = {}

    async def collect(_runtime, anchors, _conditions, *, left, right, turns, lags, offset, untouched):
        name = f"{''.join(left)}|{''.join(right)}"
        taken.setdefault(name, []).append(offset + len(anchors))
        n = len(anchors)
        return {int(lag): {key: np.zeros((n, 2)) for key in ("context", "intact", "cut", "sham_a", "sham_b")} | {"reached": np.ones((n, 1))} for lag in lags}

    def decide(slot, *, tau_seconds, seed, alpha, draws, **_design):
        name = decide.current
        lower = -0.01 if name in decide.undecided else 0.02
        estimate = SimpleNamespace(raw_rate=0.05, sham_rate=0.0)
        return estimate, 0.05, lower, 0.01

    decide.undecided = set()
    decide.current = ""

    original_advance_collect = collect

    async def collect_named(runtime, anchors, conditions, *, left, right, **kwargs):
        decide.current = f"{''.join(left)}|{''.join(right)}"
        return await original_advance_collect(runtime, anchors, conditions, left=left, right=right, **kwargs)

    monkeypatch.setattr(v25_cut, "collect_partition_samples", collect_named)
    monkeypatch.setattr(v25_cut, "decide_cut", decide)
    monkeypatch.setattr(v25_cut, "playback_decided", lambda *a, **k: False)
    taken["_decide"] = decide  # type: ignore[assignment]
    return taken


def _sweep(fail_fast: bool, stop_file: str = "") -> SweepReport:
    import asyncio

    reports = asyncio.run(
        sweep_cuts_over_lags(
            None, _anchors(16), [SimpleNamespace(name="rest")],
            lags=(3,), frame_seconds=1.0, domains=DOMAINS, looks=(8, 16), draws=10,
            fail_fast=fail_fast, stop_file=stop_file,
        )
    )
    return reports[3]


def test_a_failing_run_stops_at_the_first_undecided_cut_most_lopsided_first(scored) -> None:
    scored["_decide"].undecided = {"ABC|D"}
    report = _sweep(fail_fast=True)
    assert report.stopped_after == "ABC|D"
    assert not report.irreducible
    # The singletons come first. A decided at its first look, D was taken to its
    # last and left undecided, and nothing after it was scored.
    assert [v.name for v in report.verdicts] == ["A|BCD", "ABC|D"]
    assert scored["A|BCD"] == [8]
    assert scored["ABC|D"] == [8, 16]
    assert "ABD|C" not in scored and "AB|CD" not in scored
    assert report.as_dict()["stopped_after"] == "ABC|D"


def test_a_passing_run_reads_every_cut_as_the_full_sweep_does(scored) -> None:
    fast = _sweep(fail_fast=True)
    full = _sweep(fail_fast=False)
    assert fast.stopped_after == ""
    assert fast.irreducible and full.irreducible
    assert sorted((v.name, v.lower_bound, v.anchors_used) for v in fast.verdicts) == sorted(
        (v.name, v.lower_bound, v.anchors_used) for v in full.verdicts
    )
    assert len(fast.verdicts) == 7


def test_a_shard_stops_when_a_sibling_has_refused_the_claim(scored, tmp_path: Path) -> None:
    stop = tmp_path / "REFUSED"
    stop.write_text("C|ABD\n", encoding="utf-8")
    report = _sweep(fail_fast=True, stop_file=str(stop))
    assert report.stopped_after == "another shard"
    assert report.verdicts == []
    assert not report.irreducible


def test_the_shard_that_refuses_tells_its_siblings(scored, tmp_path: Path) -> None:
    scored["_decide"].undecided = {"A|BCD"}
    stop = tmp_path / "shards" / "REFUSED"
    _sweep(fail_fast=True, stop_file=str(stop))
    assert stop.read_text(encoding="utf-8").strip() == "A|BCD"


def test_a_merge_of_stopped_shards_is_a_refusal_and_not_an_error() -> None:
    row = {"cut": "A|BCD", "left": ["A"], "right": ["B", "C", "D"], "anchors_used": 16, "decided": False,
           "lower_bound": -0.01, "excess": 0.0, "p_value": 1.0}
    shard = {"tau_seconds": 1.0, "verdicts": [row], "stopped_after": "A|BCD", "looks": [8, 16]}
    sibling = {"tau_seconds": 1.0, "verdicts": [], "stopped_after": "another shard", "looks": [8, 16]}
    merged = merge_sweeps([shard, sibling], cuts_in_full=7)
    assert merged.stopped_after == "A|BCD"
    assert not merged.irreducible
    with pytest.raises(ValueError):
        merge_sweeps([dict(shard, stopped_after=""), dict(sibling, stopped_after="")], cuts_in_full=7)
