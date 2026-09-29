"""A forked organism does not bind the port the inter-instance listener uses.

On the four-arm campaign of 29 September the first arm to boot took the port
and the other three each recorded a bind failure, an incident and a degraded
layer at bring-up. Arms meant to differ in one switch differed in which of
their layers were online, which is a difference the design did not put there.

`quiesce_organism` has stopped this listener since a battery run was found
holding the port the live desktop uses. Stopping it after it binds is one step
too late when four organisms boot at once.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from core.consciousness.system import ConsciousnessSystem


def test_the_system_can_be_told_it_has_no_peer():
    signature = inspect.signature(ConsciousnessSystem.__init__)
    assert "offline_organism" in signature.parameters
    assert signature.parameters["offline_organism"].default is False


def test_the_subject_organism_says_it_has_no_peer():
    source = inspect.getsource(__import__("core.subject.organism", fromlist=["bring_up"]).bring_up)
    assert "offline_organism=True" in source


def test_a_live_system_still_starts_the_listener(monkeypatch):
    started: list[str] = []

    class _Server:
        _port = 8765

        async def start(self):
            started.append("bound")
            return True

        def get_status(self):
            return {}

    import core.consciousness.aura_protocol as protocol

    monkeypatch.setattr(protocol, "get_protocol_server", lambda: _Server())
    system = object.__new__(ConsciousnessSystem)
    system.offline_organism = False
    system.layer_status, system.layer_detail, system._degraded_layers = {}, {}, {}
    system._mark_layer_online = lambda name: system.layer_status.__setitem__(name, "online")
    asyncio.run(_layer_eight(system))
    assert started == ["bound"]
    assert system.layer_status["aura_protocol"] == "online"


def test_an_offline_organism_never_reaches_the_server(monkeypatch):
    started: list[str] = []

    class _Server:
        _port = 8765

        async def start(self):
            started.append("bound")
            return True

    import core.consciousness.aura_protocol as protocol

    monkeypatch.setattr(protocol, "get_protocol_server", lambda: _Server())
    system = object.__new__(ConsciousnessSystem)
    system.offline_organism = True
    system.layer_status, system.layer_detail, system._degraded_layers = {}, {}, {}
    asyncio.run(_layer_eight(system))
    assert started == []
    assert system.layer_status["aura_protocol"] == "not_started"
    # And it is not a fault: nothing asked for it.
    assert "aura_protocol" not in system._degraded_layers


async def _layer_eight(system):
    """Run only Layer 8 of `ConsciousnessSystem.start`, read from its source.

    Reading the block out of the method is how this stays honest: a test that
    reimplemented the branch would pass while the real start still bound.
    """
    import logging

    import core.consciousness.system as system_module

    source = inspect.getsource(ConsciousnessSystem.start)
    begin = source.index("# Layer 8: Aura Protocol")
    end = source.index("# \u2550\u2550\u2550", begin)
    body = inspect.cleandoc(source[begin:end])
    wrapped = "async def _run(self):\n" + "\n".join("    " + line for line in body.splitlines())
    scope = dict(vars(system_module))
    scope["logger"] = logging.getLogger("Consciousness")
    exec(compile(wrapped, "<layer8>", "exec"), scope)
    await scope["_run"](system)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
