"""A record too long to keep is made shorter, not thrown away.

Live, 2026-09-18: "what she learned about '2048-game' is too big to keep
(124122)" at the end of an eighty-eight-move run, and the next run began from
what the run before that had known. The longest list loses its older half
until the record fits, because these are records appended as she goes.
"""

from __future__ import annotations

import json

import pytest

from core.runtime import what_she_learned


@pytest.fixture
def kept_in(tmp_path, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_KEPT_IN", tmp_path)
    return tmp_path


def test_a_long_record_is_kept_with_its_newest_part(kept_in, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 2_000)
    record = {"moves": {"record": [f"pair {i}" for i in range(400)]}, "rule": "slides and combines"}
    assert what_she_learned.remember("a board", record) is True
    back = what_she_learned.recall("a board")
    assert back["rule"] == "slides and combines"
    kept = back["moves"]["record"]
    assert kept and kept[-1] == "pair 399"
    assert "pair 0" not in kept
    assert len(json.dumps(back)) <= 2_000


def test_short_lists_are_never_cut_to_make_room(kept_in, monkeypatch):
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 200)
    record = {"down_at": [0.1, 0.2, 0.3, 0.4], "note": "x" * 400}
    assert what_she_learned.remember("a board", record) is True
    back = what_she_learned.recall("a board")
    assert back["down_at"] == [0.1, 0.2, 0.3, 0.4]
    assert "note" not in back


def test_a_part_that_cannot_be_shortened_is_let_go_and_the_rest_kept(kept_in, monkeypatch, caplog):
    """Live, 26 Sep: "too big to keep (1364611)" at the end of the run that
    reached 2048, and everything else it had learned went with it."""
    monkeypatch.setattr(what_she_learned, "_MOST_KEPT", 2_000)
    record = {
        "places": {f"{x},{y}": x * y for x in range(40) for y in range(40)},
        "moves": {"right": {"slides and combines": 40}},
        "lattice": {"down_at": [0.3, 0.45, 0.6, 0.75]},
    }
    with caplog.at_level("INFO", logger=what_she_learned.logger.name):
        assert what_she_learned.remember("a board", record) is True
    back = what_she_learned.recall("a board")
    assert back["moves"] == record["moves"] and back["lattice"] == record["lattice"]
    assert "places" not in back
    assert "places (" in caplog.text, "what was let go is named"
