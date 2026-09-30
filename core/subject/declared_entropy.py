"""The entropy a subject-core run declares, in place of the one the machine has.

Two sources. Managed entropy, below. And the controlled chaos engine that
perturbs the liquid substrate on every step: its somatic noise was a hash of
the machine's uptime, its own process's thread count and memory, and the CPU's
temperature, seeded from the process id and its start time, so every process
drove C with different noise and no seed could replay it. A run declares it
from its own clock and the host reading it prepared, seeded from the run's seed.

`core.runtime.managed_entropy` feeds the predictive self model's observation
noise and agency's curiosity jitter, and it draws from the ANU quantum random
number generator over the network, then from `os.urandom`. Neither is seeded
and neither is carried by a fork, so the self model's weights took noise no
seed could reproduce: two processes on one seed drifted apart in S, and within
one process a restored anchor's turn did not repeat. The run's own Python
generator is seeded from the run's seed and carried by every snapshot, so
drawing from it makes the noise hers to replay. The injection itself, its size
and its budget, are untouched.
"""

from __future__ import annotations

import hashlib
import random
import time
from typing import Any

__all__ = ["declare_the_entropy", "release_the_entropy"]


def declare_the_entropy(runtime: Any) -> None:
    """Draw the managed entropy from the run's own generator until the run releases it."""
    try:
        from core.runtime.managed_entropy import get_managed_entropy
    except (ImportError, AttributeError):
        # not a failure: no managed entropy means nothing draws true noise.
        return
    source = get_managed_entropy()
    runtime.declared_entropy = source
    # An attribute on the instance, which the class's own method sits behind
    # and comes back from when it is removed.
    source._raw_float = random.random
    _declare_the_chaos(runtime)


def _declare_the_chaos(runtime: Any) -> None:
    try:
        from core.consciousness.controlled_chaos import declare_the_machine
    except (ImportError, AttributeError):
        # not a failure: no chaos engine means no somatic noise to declare.
        return

    def signals() -> list[float]:
        # The run's clock, and the host the turn's condition prepared: what
        # the machine's own readings stood for, declared.
        now = time.time()
        local = time.localtime(now)
        hardware = dict(getattr(getattr(runtime.state, "soma", None), "hardware", {}) or {})
        declared = [float(hardware[k]) for k in sorted(hardware) if isinstance(hardware[k], (int, float))]
        return [local.tm_hour / 24.0, local.tm_min / 60.0, (now % 60.0) / 60.0, *declared]

    seed = hashlib.sha256(f"subject-core-chaos-{int(getattr(runtime, 'seed', 0))}".encode()).digest()
    declare_the_machine(signals, seed)
    runtime.declared_chaos = True


def release_the_entropy(runtime: Any) -> None:
    """Put back the entropy source the run declared over."""
    source = getattr(runtime, "declared_entropy", None)
    if source is None:
        return
    source.__dict__.pop("_raw_float", None)
    runtime.declared_entropy = None
    if getattr(runtime, "declared_chaos", False):
        from core.consciousness.controlled_chaos import declare_the_machine

        declare_the_machine(None, None)
        runtime.declared_chaos = False
