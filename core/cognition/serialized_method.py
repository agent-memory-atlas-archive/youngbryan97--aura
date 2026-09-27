"""A method run while holding its instance's lock."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from functools import wraps
from typing import Any, Concatenate, Protocol


class _HoldsALock(Protocol):
    @property
    def _lock(self) -> AbstractContextManager[Any]: ...


def serialized[S: _HoldsALock, **P, R](
    method: Callable[Concatenate[S, P], R],
) -> Callable[Concatenate[S, P], R]:
    """Run ``method`` holding ``self._lock``, so its reads and writes are one step."""

    @wraps(method)
    def call(self: S, /, *args: P.args, **kwargs: P.kwargs) -> R:
        with self._lock:
            return method(self, *args, **kwargs)

    return call
