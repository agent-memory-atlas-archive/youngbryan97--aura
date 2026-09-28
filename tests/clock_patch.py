"""Replace a module's clock functions for that module only.

`monkeypatch.setattr(module.time, "sleep", fake)` does not patch the module.
`module.time` IS the `time` module, so the fake replaces `time.sleep` for
every thread in the process until the test ends: a flusher, a sweeper or a
watchdog left running by an earlier test sleeps through the fake too, and a
test that records its sleeps records theirs. Nineteen tests did this for
`sleep`, and one had already done it for `time.monotonic` and been found by
the order dependence it caused.

This swaps the module's own reference to `time` for a copy whose named
functions are replaced, so the code under test sees the fake and nothing
else does.
"""
from __future__ import annotations

import types
from typing import Any, Callable


class _ClockProxy(types.SimpleNamespace):
    """A module's view of `time` with some functions replaced."""


def _family(module: types.ModuleType) -> list[types.ModuleType]:
    """The module and every loaded module lifted out of it.

    The size budget lifts functions into sibling modules, each with its own
    `import time`, so the code a test means by "this module" can run from a
    sibling the test never names. The shared reading of a lift is
    `source_contract.lifted_siblings`.
    """
    import sys

    from tests.source_contract import lifted_siblings

    found = [module]
    origin = getattr(module, "__file__", None)
    if not origin:
        return found
    package = module.__name__.rpartition(".")[0]
    for path in lifted_siblings(origin):
        name = f"{package}.{path.stem}" if package else path.stem
        sibling = sys.modules.get(name)
        if sibling is not None and sibling is not module:
            found.append(sibling)
    return found


def patch_module_clock(monkeypatch: Any, module: types.ModuleType, **fakes: Callable[..., Any]) -> None:
    import time as real_time

    for member in _family(module):
        real = getattr(member, "time", None)
        if real is not real_time and not isinstance(real, _ClockProxy) and member is not module:
            continue  # a sibling with no `import time` of its own reads no clock
        # A second call on the same module adds to the first rather than
        # rebuilding from the real clock and dropping the earlier fake.
        base = real if isinstance(real, _ClockProxy) else real_time
        proxy = _ClockProxy(
            **{name: getattr(base, name) for name in dir(base) if not name.startswith("__")}
        )
        for name, fake in fakes.items():
            if not hasattr(real_time, name):
                raise AttributeError(f"time has no {name!r} to replace")
            setattr(proxy, name, fake)
        monkeypatch.setattr(member, "time", proxy)


def patch_module_sleep(monkeypatch: Any, module: types.ModuleType, fake: Callable[..., Any]) -> None:
    patch_module_clock(monkeypatch, module, sleep=fake)
