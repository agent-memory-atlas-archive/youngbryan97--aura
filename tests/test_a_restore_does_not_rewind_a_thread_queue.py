"""A restore does not rewind a queue that a thread is working through.

The fork left locks, conditions, pipes and the asyncio and multiprocessing
queues where they were, and copied everything else an organ held. The threads'
own `queue.Queue` was everything else: its lock and conditions stayed and its
counts were copied and put back. The state registry's dispatcher takes a
snapshot off its queue and marks it done once its listeners have it, so a
restore between the two rewound the count under it, and on the seed-7 run of
26 September the dispatcher died of "task_done() called too many times".
"""

from __future__ import annotations

import queue

from core.subject.copies import _is_process_furniture
from core.subject.snapshot import _organ_state, _restore_organ


class _Registry:
    def __init__(self) -> None:
        self.notify = queue.Queue()
        self.version = 0


def test_thread_queues_are_furniture():
    for held in (queue.Queue(), queue.LifoQueue(), queue.PriorityQueue(), queue.SimpleQueue()):
        assert _is_process_furniture(held), type(held)


def test_an_item_taken_before_a_restore_can_still_be_marked_done():
    registry = _Registry()
    saved = _organ_state(registry)
    registry.notify.put("snapshot")
    registry.version = 1
    assert registry.notify.get(timeout=1.0) == "snapshot"
    _restore_organ(registry, saved)
    assert registry.version == 0
    registry.notify.task_done()
