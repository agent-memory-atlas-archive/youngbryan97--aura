"""A fork snapshot written in one process restores in another, functions and all.

Shards fork from the coordinator's anchors so every cut is a cut of one
organism. A snapshot's state pickles; a lambda or closure in it does not, and
travels as a marker that the reader fills from the same place in a snapshot of
its own.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np
import pytest

from core.subject.anchor_travel import LiveFunction, dumps, fill_from, loads

pytestmark = pytest.mark.unit


class _Store:
    def __init__(self, scale: float) -> None:
        self.values = np.arange(4.0) * scale
        self.make = lambda: scale  # the kind of thing that does not pickle
        self.named = {"factory": lambda: "made"}


@dataclass
class _Snapshot:
    organs: dict = field(default_factory=dict)
    turn: int = 0


def _snapshot(scale: float, turn: int) -> _Snapshot:
    store = _Store(scale)
    return _Snapshot(organs={"store": store, "again": store}, turn=turn)


def test_state_travels_and_functions_come_from_the_reader() -> None:
    written = _snapshot(2.0, turn=7)
    travelled = loads(dumps(written))
    assert isinstance(travelled.organs["store"].make, LiveFunction)
    own = _snapshot(5.0, turn=1)
    filled = fill_from(travelled, own)
    assert filled.turn == 7 and np.array_equal(filled.organs["store"].values, np.arange(4.0) * 2.0)
    assert filled.organs["store"].make is own.organs["store"].make
    assert filled.organs["store"].named["factory"]() == "made"
    assert filled.organs["again"] is filled.organs["store"]  # one object, reached twice


def test_a_function_with_no_place_in_the_reader_is_refused() -> None:
    travelled = loads(dumps(_snapshot(2.0, turn=7)))
    with pytest.raises(ValueError, match="no live function"):
        fill_from(travelled, _Snapshot(organs={}))


def test_the_coordinators_bank_reaches_a_shard_whole(tmp_path) -> None:
    """The runner's own write and read: anchors, doses and scales, functions from the shard."""
    import asyncio
    from types import SimpleNamespace

    import tools.run_subject_core_v25 as runner
    from core.subject.v25_runtime import Anchor

    anchors = [
        Anchor(snapshot=_snapshot(float(i + 1), turn=i), current=np.full(3, float(i)), history=np.zeros(6),
               source_condition="rest")
        for i in range(3)
    ]
    bank = {"anchors": anchors, "doses": {"A": 0.5}, "scale": np.ones(3), "live_mask": np.ones(3, bool),
            "baseline_mean": np.zeros(3)}
    path = tmp_path / runner.ANCHOR_BANK
    assert runner._write_anchor_bank(path, bank) > 0
    shard = SimpleNamespace(snapshot=lambda: _snapshot(9.0, turn=0))
    read = asyncio.run(runner._read_anchor_bank(path, shard, wait_seconds=1.0))
    assert read["doses"] == {"A": 0.5}
    assert [a.snapshot.turn for a in read["anchors"]] == [0, 1, 2]
    assert np.array_equal(read["anchors"][2].current, np.full(3, 2.0))
    assert callable(read["anchors"][1].snapshot.organs["store"].make)


def test_a_lock_held_inside_the_state_is_the_readers_own() -> None:
    import threading

    written = _snapshot(2.0, turn=3)
    written.organs["store"].guard = threading.RLock()
    travelled = loads(dumps(written))
    assert isinstance(travelled.organs["store"].guard, LiveFunction)
    own = _snapshot(5.0, turn=0)
    own.organs["store"].guard = threading.RLock()
    filled = fill_from(travelled, own)
    assert filled.organs["store"].guard is own.organs["store"].guard
    assert filled.turn == 3


class _Handle:
    """A kind the list of resources has never named, which declines a pickle."""

    def __reduce_ex__(self, protocol):
        raise TypeError("cannot pickle '_Handle' object")


def test_a_kind_nobody_listed_is_the_readers_own_if_it_refuses_a_pickle() -> None:
    import json

    written = _snapshot(2.0, turn=4)
    written.organs["store"].codec = json  # a module, the kind probe 5 died on
    written.organs["store"].handle = _Handle()
    travelled = loads(dumps(written))
    assert isinstance(travelled.organs["store"].codec, LiveFunction)
    assert isinstance(travelled.organs["store"].handle, LiveFunction)
    own = _snapshot(5.0, turn=0)
    own.organs["store"].codec = json
    own.organs["store"].handle = _Handle()
    filled = fill_from(travelled, own)
    assert filled.organs["store"].codec is json
    assert filled.organs["store"].handle is own.organs["store"].handle
    assert np.array_equal(filled.organs["store"].values, np.arange(4.0) * 2.0)


class _Slotted:
    __slots__ = ("make", "count")

    def __init__(self, make, count):
        self.make = make
        self.count = count


def test_containers_that_insist_on_a_callable_load_and_are_filled_in_place() -> None:
    """Probe 6 died loading a defaultdict whose factory was a lambda."""
    import collections
    import functools

    def build(scale: float):
        shared = {"weight": scale}
        return {
            "counts": collections.defaultdict(lambda: scale, {"a": 1.0}),
            "call": functools.partial(lambda x, y: x * y + scale, scale),
            "first": shared,
            "second": shared,
            "slotted": _Slotted(lambda: scale, 3),
        }

    travelled = loads(dumps(build(2.0)))
    own = build(5.0)
    filled = fill_from(travelled, own)
    assert isinstance(filled["counts"], collections.defaultdict)
    assert filled["counts"]["a"] == 1.0
    assert filled["counts"]["b"] == 5.0  # the reader's factory
    assert filled["call"](3.0) == 2.0 * 3.0 + 5.0  # the writer's argument, the reader's function
    assert filled["first"] is filled["second"]
    assert filled["first"]["weight"] == 2.0
    assert filled["slotted"].make() == 5.0 and filled["slotted"].count == 3


def test_a_lock_the_reader_has_no_place_for_is_made_new() -> None:
    """Probe 6: the writer's list was longer than the reader's, and its extra
    entry held an RLock with nothing at the same place to take it from."""
    import threading

    written = _snapshot(2.0, turn=1)
    written.organs["extra"] = {"guard": threading.RLock(), "flag": threading.Event()}
    filled = fill_from(loads(dumps(written)), _snapshot(5.0, turn=0))
    guard = filled.organs["extra"]["guard"]
    assert type(guard) is type(threading.RLock())
    assert isinstance(filled.organs["extra"]["flag"], threading.Event)


def test_every_hole_is_listed_with_where_it_was() -> None:
    written = _snapshot(2.0, turn=1)
    written.organs["extra"] = [{"make": lambda: 1}, {"make": lambda: 2}]
    holes: list[str] = []
    fill_from(loads(dumps(written)), _snapshot(5.0, turn=0), holes=holes)
    assert len(holes) == 2
    assert holes[0].startswith(".organs.extra[0].make: ")


def test_a_module_object_is_the_readers_own_and_takes_the_writers_state() -> None:
    """`module_state` keeps the live object beside its copy. Restored as it
    travelled, the reader's module would be rebound to the writer's object."""
    import threading

    class _Emitter:
        def __init__(self, sent: int) -> None:
            self.sent = sent
            self.lock = threading.Lock()

    written = _Snapshot()
    written.module_state = {"core.thought_stream:_emitter": (_Emitter(9), {"sent": 9})}
    own = _Snapshot()
    reader_emitter = _Emitter(2)
    own.module_state = {"core.thought_stream:_emitter": (reader_emitter, {"sent": 2})}
    from core.subject.anchor_travel import fill_snapshot

    filled = fill_snapshot(loads(dumps(written)), own)
    held, captured = filled.module_state["core.thought_stream:_emitter"]
    assert held is reader_emitter and held.sent == 2  # the object is untouched until the restore
    assert captured == {"sent": 9}  # and what it will be given is the writer's


def test_a_loop_is_the_readers_running_loop() -> None:
    import asyncio

    async def inside() -> None:
        written = _snapshot(2.0, turn=1)
        written.organs["extra"] = {"loop": asyncio.get_running_loop()}
        filled = fill_from(loads(dumps(written)), _snapshot(5.0, turn=0))
        assert filled.organs["extra"]["loop"] is asyncio.get_running_loop()

    asyncio.run(inside())


def test_the_writers_state_files_are_restored_under_the_readers_root() -> None:
    """Probe 7: the restore refuses stores saved under another root, so the
    reader kept its own memories, beliefs and goals, and 93 to 115 columns
    differed per anchor. The saved paths move to the reader's root."""
    from pathlib import Path

    from core.subject.anchor_travel import fill_snapshot

    writer, reader = "/runs/writer/state", "/runs/reader/state"
    written = _snapshot(2.0, turn=1)
    written.stores = {
        "root": writer,
        "places": [(writer, True, True)],
        "entries": {f"{writer}/memory.db": ("database", (1, 2), b"rows")},
        "directories": [f"{writer}/beliefs"],
    }
    written.module_state = {}
    written.organs["store"].path = Path(f"{writer}/goals.json")
    written.organs["store"].elsewhere = "/runs/writer/state2/not-under-the-root"
    own = _snapshot(5.0, turn=0)
    own.stores = {"root": reader, "places": [], "entries": {}, "directories": []}
    own.module_state = {}
    filled = fill_snapshot(loads(dumps(written)), own)
    assert filled.stores["root"] == reader
    assert filled.stores["places"] == [(reader, True, True)]
    assert list(filled.stores["entries"]) == [f"{reader}/memory.db"]
    assert filled.stores["entries"][f"{reader}/memory.db"][2] == b"rows"
    assert filled.stores["directories"] == [f"{reader}/beliefs"]
    assert filled.organs["store"].path == Path(f"{reader}/goals.json")
    assert filled.organs["store"].elsewhere == "/runs/writer/state2/not-under-the-root"


def test_a_handle_on_the_writers_own_process_is_the_readers() -> None:
    """The metabolic monitor read the writer's exited pid all through probe 7."""
    import os

    from core.runtime.resource_psutil import Process

    written = _snapshot(2.0, turn=1)
    written.organs["store"].me = Process()
    written.organs["store"].other = Process(1)
    travelled = loads(dumps(written))
    assert isinstance(travelled.organs["store"].me, LiveFunction)
    assert travelled.organs["store"].other.pid == 1  # someone else's process is state
    filled = fill_from(travelled, _snapshot(5.0, turn=0))
    assert isinstance(filled.organs["store"].me, Process) and filled.organs["store"].me.pid == os.getpid()


class _OneOfMe:
    """Hands out one instance, as the thought emitter does."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.sent = 0
        return cls._instance


def test_a_singleton_travels_as_the_readers_and_the_readers_is_not_written_over() -> None:
    """Probe 7: unpickling the writer's emitter returned the reader's live one
    and wrote the writer's markers over it; its lock became a marker."""
    import threading

    mine = _OneOfMe()
    mine.sent = 5
    mine.lock = threading.Lock()
    written = _snapshot(2.0, turn=1)
    written.organs["store"].emitter = mine
    data = dumps(written)
    before = dict(vars(mine))
    travelled = loads(data)
    assert vars(mine) == before  # loading did not touch the live one
    filled = fill_from(travelled, _snapshot(5.0, turn=0))
    assert filled.organs["store"].emitter is _OneOfMe._instance


class _Kind(enum.Enum):
    SOCIAL = "social"


def test_an_enum_member_travels_as_itself_and_can_be_a_key() -> None:
    """Probe 8: members held by their class travelled as singleton markers, and
    a dict keyed by them lost every value under the key."""
    written = _snapshot(2.0, turn=1)
    written.organs["store"].adapters = {_Kind.SOCIAL: lambda: "reader's own"}
    own = _snapshot(5.0, turn=0)
    own.organs["store"].adapters = {_Kind.SOCIAL: lambda: "reader's own"}
    filled = fill_from(loads(dumps(written)), own)
    assert list(filled.organs["store"].adapters) == [_Kind.SOCIAL]
    assert filled.organs["store"].adapters[_Kind.SOCIAL]() == "reader's own"
