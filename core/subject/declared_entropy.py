"""The entropy a subject-core run declares, in place of the one the machine has.

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

import random
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


def release_the_entropy(runtime: Any) -> None:
    """Put back the entropy source the run declared over."""
    source = getattr(runtime, "declared_entropy", None)
    if source is None:
        return
    source.__dict__.pop("_raw_float", None)
    runtime.declared_entropy = None
