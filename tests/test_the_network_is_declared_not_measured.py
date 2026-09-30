"""A subject-core run declares its network; it does not measure it.

Each turn writes the connectivity status into her world facts, and the status
was a TCP connect to a public resolver with its latency in milliseconds. Two
processes restored from one anchor on 29 September differed in nothing but
that fact: 20.57 ms in one and 21.14 in the other, which moved her world
domain's fact profile by up to 0.078. The run declares the host it lives on,
and now the network too.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from core.runtime.connectivity import get_connectivity_status
from core.subject.driver import install_declared_host, release_declared_host

pytestmark = pytest.mark.unit


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace(soma=SimpleNamespace(hardware={})), declared_host=None)


def test_a_declared_run_reads_the_network_it_declared(monkeypatch) -> None:
    def refuse(*args, **kwargs):
        raise AssertionError("a declared network opened a socket")

    monkeypatch.setattr(socket, "create_connection", refuse)
    runtime = _runtime()
    install_declared_host(runtime)
    try:
        status = get_connectivity_status(force=True)
        assert status.mode == "declared"
        assert status.online is True
        assert status.latency_ms is None
    finally:
        release_declared_host(runtime)


def test_releasing_the_host_puts_the_measuring_probe_back() -> None:
    from core.runtime.connectivity import ConnectivityProbe, get_connectivity_probe

    runtime = _runtime()
    install_declared_host(runtime)
    assert not isinstance(get_connectivity_probe(), ConnectivityProbe)
    release_declared_host(runtime)
    assert isinstance(get_connectivity_probe(), ConnectivityProbe)


def test_two_readings_at_one_moment_are_the_same_reading(monkeypatch) -> None:
    """What makes two processes at one anchor agree: nothing in it but the clock."""
    import time

    monkeypatch.setattr(time, "time", lambda: 1_790_000_000.0)
    runtime = _runtime()
    install_declared_host(runtime)
    try:
        assert get_connectivity_status().to_dict() == get_connectivity_status(force=True).to_dict()
    finally:
        release_declared_host(runtime)
