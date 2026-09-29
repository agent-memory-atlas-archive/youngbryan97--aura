"""The null table is the same numbers whether one process computes it or several.

Each synthetic architecture is its own toy system built from its own seed, so
its row cannot depend on which process built it or in what order. The null
stage of a seed-7 campaign took 4 h 35 min in one process; `--null-workers`
spreads the rows across processes, and that is only honest if the table does
not move.
"""

from __future__ import annotations

import pytest

from core.subject import null_table


@pytest.fixture
def cheap(monkeypatch):
    """A row that costs nothing, so the ordering and the pooling are what is tested."""

    def row(name, *, seed, null_draws, trials, domains):
        return {"phi_do": float(len(name) + seed + null_draws + trials), "domains": list(domains)}, [f"{name} scored"]

    monkeypatch.setattr(null_table, "null_row", row)


def test_rows_come_back_in_the_order_asked(cheap) -> None:
    names = ["star", "hub", "recurrent", "ring"]
    out = null_table.null_rows(names, seed=7, null_draws=2, trials=3, domains=("A", "B"), workers=1)
    assert [name for name, _row, _notes in out] == names
    assert out[1][1]["phi_do"] == 3 + 7 + 2 + 3
    assert out[0][2] == ["star scored"]


@pytest.mark.slow
def test_one_process_and_several_give_the_same_table() -> None:
    from core.subject.nulls import ARCHITECTURES
    from core.subject.state import DOMAINS

    names = list(ARCHITECTURES)[:2]
    one = null_table.null_rows(names, seed=7, null_draws=1, trials=1, domains=DOMAINS, workers=1)
    many = null_table.null_rows(names, seed=7, null_draws=1, trials=1, domains=DOMAINS, workers=2)
    assert one == many
