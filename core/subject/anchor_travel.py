"""Fork snapshots, written by one process and restored in another.

Two processes on one seed come up as two organisms, so shards that collect
their own anchors sweep different organisms, and their merge is refused. If the
shards fork from the coordinator's anchors instead, every cut is a cut of one
organism. A snapshot holds her state as copies, which pickle, and a few
functions, which may not: `deepcopy` passes a function through by reference,
and a lambda or a closure has no name another process could import it by (the
first one met was a store factory in `IntentionalRetriever`).

A function is behaviour, not state, and the reading process has the same
functions in the same places. So an unimportable function travels as a marker
naming where it was, and the reader fills each marker from the same place in a
snapshot of its own before it restores. Everything that is state still comes
from the writer.
"""

from __future__ import annotations

import asyncio
import collections
import dataclasses
import enum
import functools
import io
import os
import pickle
import types
from pathlib import PurePath
from typing import Any

__all__ = ["LiveFunction", "dumps", "fill_from", "fill_snapshot", "loads"]


@dataclasses.dataclass(frozen=True)
class LiveFunction:
    """A function, or a thing of a class made at runtime, that did not travel: the reader's own is used."""

    qualname: str
    #: For a lock, an event, a queue: the kind, so a fresh one can stand in
    #: where the reader has none at the same place.
    fresh: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Callable so a container that insists on one loads: a defaultdict's
        # factory, a partial's function. Never called: filled before a restore.
        raise RuntimeError(f"{self.qualname} was not filled from the reader before use")


def _importable(function: Any) -> bool:
    qualname = getattr(function, "__qualname__", "")
    return bool(qualname) and "<locals>" not in qualname and "<lambda>" not in qualname


def _infrastructure() -> tuple[type, ...]:
    """Kinds of object that hold a process's resources rather than her state."""
    import asyncio
    import concurrent.futures
    import io as _io
    import socket
    import sqlite3
    import contextvars
    import hashlib
    import mmap
    import queue
    import selectors
    import ssl
    import subprocess
    import threading
    import weakref

    kinds: list[type] = [
        type(threading.Lock()), type(threading.RLock()), threading.Condition, threading.Event,
        threading.Semaphore, threading.Thread, threading.Barrier, type(threading.local()),
        sqlite3.Connection, sqlite3.Cursor, socket.socket, _io.IOBase,
        asyncio.AbstractEventLoop, asyncio.Future, concurrent.futures.Executor, concurrent.futures.Future,
        types.GeneratorType, types.CoroutineType, types.AsyncGeneratorType, weakref.ref,
        queue.SimpleQueue, selectors.BaseSelector, subprocess.Popen, memoryview, mmap.mmap,
        contextvars.ContextVar, contextvars.Context, types.FrameType, types.TracebackType,
        type(hashlib.sha256()), ssl.SSLContext,
    ]
    try:
        from multiprocessing.connection import Connection

        kinds.append(Connection)
    except ImportError:  # not a failure: a build without multiprocessing has none to carry
        pass
    return tuple(kinds)


_INFRASTRUCTURE = _infrastructure()


def _fresh_kinds() -> dict[type, tuple[str, Any]]:
    """Resources with no state worth carrying, which a new one replaces exactly."""
    import queue
    import threading

    return {
        type(threading.Lock()): ("lock", threading.Lock),
        type(threading.RLock()): ("rlock", threading.RLock),
        threading.Condition: ("condition", threading.Condition),
        threading.Event: ("event", threading.Event),
        threading.Semaphore: ("semaphore", threading.Semaphore),
        threading.BoundedSemaphore: ("bounded-semaphore", threading.BoundedSemaphore),
        threading.local: ("local", threading.local),
        queue.SimpleQueue: ("simple-queue", queue.SimpleQueue),
    }


_FRESH = _fresh_kinds()
_FRESH_BY_NAME = {name: make for name, make in _FRESH.values()}
# The loop an object was bound to in the writer is the reader's running loop
# here: there is one, and the restore runs on it.
_FRESH_BY_NAME["event-loop"] = asyncio.get_running_loop
#: Modules whose `Process` takes no argument to mean the current process.
_PROCESS_HANDLES = frozenset({"psutil", "core.runtime.resource_psutil"})


def _made_fresh(name: str) -> Any:
    import importlib

    if name.startswith("this-process:"):
        _, module, qualname = name.split(":", 2)
        return getattr(importlib.import_module(module), qualname)()
    if name.startswith("singleton:"):
        _, module, qualname, attribute = name.split(":", 3)
        kind: Any = importlib.import_module(module)
        for part in qualname.split("."):
            kind = getattr(kind, part)
        # The reader's own singleton; if it has none yet, the one its class makes.
        return getattr(kind, attribute, None) or kind()
    return _FRESH_BY_NAME[name]()

# What the pickler writes itself without asking the object: `persistent_id` must
# not ask these to reduce, because a function or a PickleBuffer refuses a
# reduce and still pickles.
_WRITTEN_DIRECTLY = frozenset({
    type(None), bool, int, float, complex, str, bytes, bytearray, tuple, list, dict, set, frozenset,
    types.FunctionType, types.BuiltinFunctionType, type, pickle.PickleBuffer,
})
_REFUSES: dict[type, bool] = {}


def _refuses_to_reduce(obj: Any) -> bool:
    """Whether a thing of this kind declines to be pickled at all: a module, a lock, a handle.

    The list above names the kinds met so far; this catches the next one, asked
    once per kind. It asks only the object itself, so what it holds is still
    judged one piece at a time.
    """
    kind = type(obj)
    known = _REFUSES.get(kind)
    if known is None:
        try:
            obj.__reduce_ex__(pickle.HIGHEST_PROTOCOL)
            known = False
        except TypeError:
            known = True
        _REFUSES[kind] = known
    return known


def _held_by_its_class(kind: type) -> dict[int, str]:
    """Objects a class keeps as its own attributes, by id: its singletons.

    A class that hands out one instance from `__new__` hands the reader's out
    again when a pickle asks for a new one, and the writer's state is then
    written over the reader's live object before anything can fill it. The
    thought emitter did that in probe 7, and its lock became a marker.
    """
    held: dict[int, str] = {}
    for klass in kind.__mro__:
        if klass.__module__ == "builtins":
            continue
        for name, value in list(vars(klass).items()):
            if isinstance(value, kind):
                held[id(value)] = name
    return held


class _Writer(pickle.Pickler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._singletons: dict[type, dict[int, str]] = {}

    def persistent_id(self, obj: Any) -> Any:
        kind = type(obj)
        # An enum member is held by its class too, and pickles by name into
        # the reader's own member already; as a marker it could not be a key.
        if kind.__module__ not in ("builtins", "numpy") and not isinstance(obj, (type, enum.Enum)):
            held = self._singletons.get(kind)
            if held is None:
                held = self._singletons[kind] = _held_by_its_class(kind)
            name = held.get(id(obj))
            if name is not None:
                return ("resource", f"singleton:{kind.__module__}:{kind.__qualname__}:{name}")
        if isinstance(obj, (types.FunctionType, types.LambdaType)) and not _importable(obj):
            return ("live-function", getattr(obj, "__qualname__", "?"))
        if isinstance(obj, types.MethodType) and not _importable(obj.__func__):
            return ("live-function", getattr(obj.__func__, "__qualname__", "?"))
        # A class made inside a function, and anything made from one: a model
        # built from a schema at runtime (`_a_model_from_a_schema`). The reader
        # builds the same one the same way, and has it in the same place.
        if isinstance(obj, type) and not _importable(obj):
            return ("live-function", obj.__qualname__)
        if not isinstance(obj, type) and not _importable(type(obj)) and type(obj).__module__ != "builtins":
            return ("live-function", type(obj).__qualname__)
        # And a lock, a connection, a thread, a file held by reference inside
        # something copied: a process's resources, not her state. The reader's
        # own at the same place is the right one, and a lock or an event can
        # be made new where the reader has none there.
        fresh = _FRESH.get(type(obj))
        if fresh is not None:
            return ("resource", fresh[0])
        if isinstance(obj, asyncio.AbstractEventLoop):
            return ("resource", "event-loop")
        # A handle on the writer's own process watches a process that has
        # exited by the time the reader restores it (the metabolic monitor
        # read a dead pid all through probe 7). The reader's own is meant.
        if type(obj).__name__ == "Process" and getattr(obj, "pid", None) == os.getpid():
            kind = type(obj)
            if kind.__module__ in _PROCESS_HANDLES:
                return ("resource", f"this-process:{kind.__module__}:{kind.__qualname__}")
        if isinstance(obj, _INFRASTRUCTURE):
            return ("live-function", type(obj).__qualname__)
        if type(obj) not in _WRITTEN_DIRECTLY and not isinstance(obj, type) and _refuses_to_reduce(obj):
            return ("live-function", type(obj).__qualname__)
        return None


class _Reader(pickle.Unpickler):
    def persistent_load(self, pid: Any) -> Any:
        kind, qualname = pid
        if kind == "resource":
            return LiveFunction(qualname, fresh=qualname)
        if kind != "live-function":
            raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")
        return LiveFunction(qualname)


def dumps(value: Any) -> bytes:
    buffer = io.BytesIO()
    _Writer(buffer, protocol=pickle.HIGHEST_PROTOCOL).dump(value)
    return buffer.getvalue()


def loads(data: bytes) -> Any:
    return _Reader(io.BytesIO(data)).load()


def fill_snapshot(travelled: Any, own: Any, *, holes: list[str] | None = None) -> Any:
    """A fork snapshot written in another process, made ready to restore in this one.

    A snapshot keeps each module-level object beside a copy of what it held,
    `(held, captured)`, so a restore can put the object back under its name.
    Written elsewhere, `held` is a copy of the writer's object, and restoring
    it would rebind the reader's module to it, loop, locks and wiring
    included. So the reader's own object under the same name is kept, and the
    writer's copy of what it held is restored into it. A name the reader never
    filled keeps the object that travelled.
    
    Every path under the writer's state root is moved to the reader's. The
    restore refuses stores saved under another root, rightly, since writing
    them would put one run's files in another's; the reader has its own root,
    and without the move its memories, beliefs and goals stayed its own
    (probe 7: 93 to 115 columns apart per anchor, most of them content).
    """
    mine = getattr(own, "module_state", None) or {}
    theirs = getattr(travelled, "module_state", None)
    kept: list[Any] = []
    if isinstance(theirs, dict):
        for key, pair in list(theirs.items()):
            if key in mine:
                theirs[key] = (mine[key][0], pair[1])
                kept.append(mine[key][0])
    rebase = None
    theirs_root = (getattr(travelled, "stores", None) or {}).get("root")
    mine_root = (getattr(own, "stores", None) or {}).get("root")
    if theirs_root and mine_root and theirs_root != mine_root:
        rebase = (str(theirs_root), str(mine_root))
    return fill_from(travelled, own, holes=holes, keep=kept, rebase=rebase)


def fill_from(
    travelled: Any,
    own: Any,
    *,
    holes: list[str] | None = None,
    keep: Any = (),
    rebase: tuple[str, str] | None = None,
) -> Any:
    """`travelled` with every LiveFunction replaced by what sits at the same place in `own`.

    Walks dicts by key, lists and tuples by index, and objects by their
    attributes, each object once however many paths reach it. Containers are
    filled in place, so a dict reached by two paths is still one dict and a
    defaultdict is still a defaultdict. A lock or an event with no counterpart
    is made new. Any other marker with no counterpart is a hole: an error, or,
    when `holes` is given, its path is added there and the marker stays, so one
    reading lists every hole at once. Restoring a snapshot with a hole in it
    would run a different organism than the one written. Anything in `keep`
    is the reader's own and is taken as it is. With `rebase`, a string or path
    under its first directory is moved under its second, in keys as well.
    """
    filler = _Filler(holes, rebase)
    for value in keep:
        filler.seen[id(value)] = value
    return filler.fill(travelled, own, None)


class _Filler:
    def __init__(self, holes: list[str] | None, rebase: tuple[str, str] | None = None) -> None:
        self.holes = holes
        self.seen: dict[int, Any] = {}
        self.rebase = rebase

    def _moved(self, value: Any) -> Any:
        """A path under the writer's root, as the same path under the reader's."""
        if self.rebase is None:
            return value
        old, new = self.rebase
        if isinstance(value, str):
            if value == old or value.startswith(old + os.sep):
                return new + value[len(old):]
            return value
        if isinstance(value, PurePath):
            text = str(value)
            if text == old or text.startswith(old + os.sep):
                return type(value)(new + text[len(old):])
        return value

    def _hole(self, marker: LiveFunction, path: Any) -> LiveFunction:
        where = _render(path)
        if self.holes is None:
            raise ValueError(f"no live function in the reader where {marker.qualname} was, at {where}")
        self.holes.append(f"{where}: {marker.qualname}")
        return marker

    def fill(self, travelled: Any, own: Any, path: Any) -> Any:
        seen = self.seen
        if id(travelled) in seen:
            return seen[id(travelled)]
        if isinstance(travelled, (str, PurePath)):
            return self._moved(travelled)
        if isinstance(travelled, (bytes, int, float, bool, type(None))) or type(travelled).__module__ == "numpy":
            return travelled
        if isinstance(travelled, LiveFunction):
            if own is not None and not isinstance(own, LiveFunction):
                return own
            if travelled.fresh:
                return _made_fresh(travelled.fresh)
            return self._hole(travelled, path)
        if isinstance(travelled, tuple):
            mine = own if isinstance(own, (list, tuple)) else ()
            items = [
                self.fill(value, mine[i] if i < len(mine) else None, (path, i))
                for i, value in enumerate(travelled)
            ]
            if all(a is b for a, b in zip(items, travelled)):
                filled = travelled
            elif hasattr(travelled, "_fields"):
                filled = type(travelled)._make(items)
            else:
                filled = type(travelled)(items)
            seen[id(travelled)] = filled
            return filled
        if isinstance(travelled, functools.partial):
            func = self.fill(travelled.func, getattr(own, "func", None), (path, "func"))
            args = self.fill(travelled.args, getattr(own, "args", None), (path, "args"))
            keywords = self.fill(dict(travelled.keywords), getattr(own, "keywords", None), (path, "keywords"))
            filled = functools.partial(func, *args, **keywords)
            seen[id(travelled)] = filled
            return filled
        seen[id(travelled)] = travelled
        if isinstance(travelled, dict):
            mine = own if isinstance(own, dict) else {}
            if isinstance(travelled, collections.defaultdict) and isinstance(travelled.default_factory, LiveFunction):
                travelled.default_factory = self.fill(
                    travelled.default_factory, getattr(own, "default_factory", None), (path, "default_factory")
                )
            items = list(travelled.items())
            moved = [self._moved(key) for key, _ in items]
            if any(a is not b for a, b in zip(moved, (key for key, _ in items))):
                # Rebuilt in order, so a dict keyed by path keeps its order.
                travelled.clear()
                for key, (old_key, value) in zip(moved, items):
                    travelled[key] = self.fill(value, mine.get(key, mine.get(old_key)), (path, key))
                return travelled
            for key, value in items:
                travelled[key] = self.fill(value, mine.get(key), (path, key))
            return travelled
        if isinstance(travelled, list):
            mine = own if isinstance(own, (list, tuple)) else []
            for i, value in enumerate(travelled):
                travelled[i] = self.fill(value, mine[i] if i < len(mine) else None, (path, i))
            return travelled
        if isinstance(travelled, (set, frozenset)):
            for value in travelled:
                if isinstance(value, LiveFunction):
                    self._hole(value, (path, "{}"))
            moved = {self._moved(value) for value in travelled}
            if moved == travelled:
                return travelled
            if isinstance(travelled, frozenset):
                filled = type(travelled)(moved)
                seen[id(travelled)] = filled
                return filled
            travelled.clear()
            travelled.update(moved)
            return travelled
        if isinstance(travelled, type):
            return travelled
        holder = getattr(travelled, "__dict__", None)
        if isinstance(holder, dict) and holder:
            mine = getattr(own, "__dict__", {}) if own is not None else {}
            for name, value in list(holder.items()):
                holder[name] = self.fill(value, mine.get(name), (path, name))
        for name in _slots(type(travelled)):
            if hasattr(travelled, name):
                value = self.fill(getattr(travelled, name), getattr(own, name, None), (path, name))
                object.__setattr__(travelled, name, value)
        return travelled


def _render(path: Any) -> str:
    parts: list[str] = []
    while path is not None:
        path, key = path
        parts.append(f"[{key!r}]" if not isinstance(key, str) or not key.isidentifier() else f".{key}")
    return "".join(reversed(parts)) or "<root>"


def _slots(kind: type) -> tuple[str, ...]:
    names: list[str] = []
    for klass in kind.__mro__:
        declared = klass.__dict__.get("__slots__", ())
        names.extend((declared,) if isinstance(declared, str) else declared)
    return tuple(n for n in names if n not in ("__dict__", "__weakref__"))
