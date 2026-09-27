"""The guard resolved a name with no deadline, inside a decision a turn waits on.

`socket.gethostbyname` takes no timeout and `socket.setdefaulttimeout` does not
reach it, so with no network on macOS it sits in the resolver for as long as the
resolver wants. `can_call_tool` runs it whenever she considers a tool carrying a
URL, and a turn is waiting on the answer.

Unresolvable already meant "block the URL", which is the safe direction. So a
name that will not answer quickly is one that will not answer, and the guard says
so rather than waiting.
"""

from __future__ import annotations

import time

import pytest

from core.middleware import capability_guard as guard_module


@pytest.fixture
def guard():
    made = guard_module.CapabilityGuard()
    made.capabilities = {"network": {"allowed_domains": ["*"]}}
    return made


def test_a_resolver_that_never_answers_does_not_hold_the_turn(monkeypatch, guard):
    def _never(_domain):
        time.sleep(30.0)

    monkeypatch.setattr(guard_module.socket, "gethostbyname", _never)
    monkeypatch.setattr(guard_module, "_RESOLVE_DEADLINE_S", 0.2)

    started = time.monotonic()
    allowed = guard.can_call_tool("read_url_content", {"url": "https://example.invalid/x"})
    took = time.monotonic() - started

    assert allowed is False, "a name that will not answer is blocked, not awaited"
    assert took < 5.0, f"the guard waited {took:.1f}s on a resolver"


def test_a_name_that_does_not_resolve_is_blocked(monkeypatch, guard):
    def _fails(_domain):
        raise OSError("nodename nor servname provided, or not known")

    monkeypatch.setattr(guard_module.socket, "gethostbyname", _fails)
    assert guard.can_call_tool("read_url_content", {"url": "https://nowhere.invalid/x"}) is False


def test_a_public_address_is_still_allowed(monkeypatch, guard):
    monkeypatch.setattr(guard_module.socket, "gethostbyname", lambda _d: "93.184.216.34")
    assert guard.can_call_tool("read_url_content", {"url": "https://example.com/x"}) is True


def test_a_private_address_is_still_blocked(monkeypatch, guard):
    monkeypatch.setattr(guard_module.socket, "gethostbyname", lambda _d: "127.0.0.1")
    assert guard.can_call_tool("read_url_content", {"url": "https://localhost.example/x"}) is False


def test_an_address_that_is_not_an_address_is_blocked(monkeypatch, guard):
    monkeypatch.setattr(guard_module.socket, "gethostbyname", lambda _d: "not-an-address")
    assert guard.can_call_tool("read_url_content", {"url": "https://example.com/x"}) is False


def test_the_deadline_is_declared_rather_than_left_to_the_resolver():
    assert isinstance(guard_module._RESOLVE_DEADLINE_S, float)
    assert 0.0 < guard_module._RESOLVE_DEADLINE_S <= 5.0


def test_resolves_to_returns_nothing_for_both_kinds_of_no(monkeypatch):
    def _fails(_domain):
        raise OSError("down")

    monkeypatch.setattr(guard_module.socket, "gethostbyname", _fails)
    assert guard_module._resolves_to("example.invalid") is None

    monkeypatch.setattr(guard_module.socket, "gethostbyname", lambda _d: time.sleep(30.0))
    monkeypatch.setattr(guard_module, "_RESOLVE_DEADLINE_S", 0.1)
    assert guard_module._resolves_to("example.invalid") is None
