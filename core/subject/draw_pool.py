"""Bootstrap and permutation draws across processes, in the order one process takes them.

A cut's decision cost 7.2 seconds of estimator per horizon at 32 anchors on
this host, and 22.5 at 128: a thousand bootstrap draws, each a cross-fitted
estimate, one after another on the one core its shard runs on, while most of
the machine's other cores waited.
The draws are independent once their resampled indices are drawn, so the
parent draws every index from its generator in order, the pool evaluates the
draws in contiguous chunks, and the values come back in draw order. The numbers
are the ones a single process computes; only the wall clock changes.

`AURA_ESTIMATOR_WORKERS` sets the pool's size. Unset or 1, nothing is spawned
and every draw runs inline, as before.
"""

from __future__ import annotations

import atexit
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from core.runtime.flags import env_str

__all__ = ["estimator_workers", "evaluate_in_order"]

_POOL: ProcessPoolExecutor | None = None
_POOL_SIZE = 0


def estimator_workers() -> int:
    """How many processes the draws may use, from the environment. At least one."""
    raw = env_str(
        "AURA_ESTIMATOR_WORKERS",
        description="Processes the estimator's draws may use; unset or 1 spawns none.",
        owner="core.subject.draw_pool",
    ).strip()
    try:
        return max(1, int(raw)) if raw else 1
    except ValueError:
        return 1


def _pool(size: int) -> ProcessPoolExecutor:
    """One pool per process, kept for its life: spawning costs seconds a worker."""
    global _POOL, _POOL_SIZE
    # A pool whose worker died stays broken for good, so it is replaced. The
    # test suite reaps the processes a test leaves behind, and a host under
    # memory pressure can take one.
    if _POOL is None or _POOL_SIZE != size or getattr(_POOL, "_broken", False):
        if _POOL is not None:
            _POOL.shutdown(wait=True, cancel_futures=True)
        import multiprocessing

        # Spawned rather than forked: the parent is an organism with threads,
        # and a worker needs numpy and scikit-learn, not her.
        _POOL = ProcessPoolExecutor(max_workers=size, mp_context=multiprocessing.get_context("spawn"))
        _POOL_SIZE = size
        atexit.register(_POOL.shutdown, wait=False, cancel_futures=True)
    return _POOL


def evaluate_in_order(
    chunk: Callable[[Any, Sequence[Any]], list[float]],
    shared: Any,
    draws: Sequence[Any],
    *,
    workers: int | None = None,
) -> list[float]:
    """`chunk(shared, draws[i:j])` over contiguous slices, concatenated in draw order.

    `chunk` must be a module-level function, so a spawned worker can import it,
    and it must compute each draw from that draw's own arguments alone.
    """
    size = estimator_workers() if workers is None else max(1, int(workers))
    if size <= 1 or len(draws) < 2 * size:
        return list(chunk(shared, draws))
    step = -(-len(draws) // size)
    slices = [draws[start:start + step] for start in range(0, len(draws), step)]
    from concurrent.futures.process import BrokenProcessPool

    for attempt in range(2):
        pool = _pool(size)
        try:
            futures = [pool.submit(chunk, shared, part) for part in slices]
            values: list[float] = []
            for future in futures:
                values.extend(future.result())
            return values
        except BrokenProcessPool:
            if attempt:
                raise
    raise AssertionError("unreachable")
