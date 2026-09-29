"""The network a subject-core run declares, in place of the one the machine has.

See `_DeclaredNetwork`. Installed beside the declared host
(`core.subject.driver.install_declared_host`) and released with it.
"""

from __future__ import annotations

import time
from typing import Any

__all__ = ["declare_the_network", "release_the_network"]


class _DeclaredNetwork:
    """The network a run declares: online, with no measured latency.

    Each turn writes the connectivity status into her world facts, and the
    status was a TCP connect to a public resolver with its latency in
    milliseconds. So the machine's network was read into her world domain every
    turn: the fact profile of two processes restored from one anchor differed
    by up to 0.086 on 29 September, and within a run it moved with whatever
    else the network was doing. How long a packet took is the harness's
    weather, not anything she did.
    """

    def __init__(self, target: str) -> None:
        self.target = target

    def status(self, *, force: bool = False) -> Any:
        from core.runtime.connectivity import ConnectivityStatus

        return ConnectivityStatus(
            checked_at=time.time(),
            online=True,
            mode="declared",
            target=self.target,
            reason="declared by the run",
        )


def declare_the_network(runtime: Any) -> None:
    """Stand a declared network in for the measuring probe until the run releases it."""
    try:
        from core.runtime.connectivity import (
            get_connectivity_probe,
            set_connectivity_probe_for_test,
        )
    except (ImportError, AttributeError):
        # not a failure: no connectivity module means nothing measures the
        # network, so there is nothing to declare over.
        return
    measured = get_connectivity_probe()
    target = f"{getattr(measured, 'target', 'unknown')}:{getattr(measured, 'port', '')}".rstrip(":")
    runtime.previous_network = set_connectivity_probe_for_test(_DeclaredNetwork(target))


def release_the_network(runtime: Any) -> None:
    """Put back the probe that measured the network before the run declared it."""
    try:
        from core.runtime.connectivity import set_connectivity_probe_for_test
    except (ImportError, AttributeError):
        # not a failure: the install took the same path and declared nothing.
        return
    set_connectivity_probe_for_test(getattr(runtime, "previous_network", None))
    runtime.previous_network = None
