"""Ratchet: no NEW outbound HTTP in ``core/`` outside the network gateway.

``core/runtime/network_gateway.py`` is where an outbound request meets
governance, the defensive preflight, web-content provenance, and — since
the egress privacy boundary landed — the only read of what is actually
inside the body. All of that is worth exactly as much as the share of
traffic that goes through it.

It was not all of it. Four modules held their own ``aiohttp.ClientSession``
and reached the network directly: peer belief and drive-state broadcast
(``core/collective/belief_sync.py``, four call paths, carrying Aura's own
beliefs to addresses that arrive from discovery), web search
(``core/agency/tool_orchestrator.py``, carrying the user's words to the open
web), a localhost health probe, and a shared session in the API adapter.
None of them were wrong about anything except the one thing that mattered:
they were outside the boundary, so the boundary was a boundary with doors in
it.

A convention cannot hold this. Each of those sites was written by someone
who simply reached for the HTTP client they knew, and the next one will be
too. This is the gate instead.

If this test fails on code you just wrote: call
``get_network_gateway().request()`` / ``.request_async()`` instead of an
HTTP client directly. Vendor model SDKs are not an exception: remote model
providers have been removed from Aura's runtime.
"""
from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Calls that put bytes on a socket without passing the gateway.
_DIRECT_EGRESS_CALLS = {
    "ClientSession",  # aiohttp
    "urlopen",  # urllib.request
    "build_opener",
    "HTTPConnection",  # http.client
    "HTTPSConnection",
}

#: Clients built first and used afterwards. `httpx.get(...)` was caught and
#: `httpx.AsyncClient().post(...)` was not, because the verb sits on an
#: instance rather than on the library; the constructor is where it shows.
_CLIENT_CONSTRUCTORS = {
    "httpx": {"Client", "AsyncClient"},
    "requests": {"Session"},
    "websockets": {"connect"},
    "websocket": {"create_connection"},
}

#: Every package that ships, not only core/. A skill that sends a request
#: carries the person's words as surely as core does, and the scan used to
#: stop at the core/ directory.
_PRODUCTION = ("core", "interface", "skills", "llm", "executors", "security")

#: Module attributes that are an HTTP verb on a client library.
_DIRECT_EGRESS_ROOTS = {"httpx", "requests", "urllib3", "aiohttp"}
_HTTP_VERBS = {"get", "post", "put", "patch", "delete", "head", "options", "stream", "request"}

#: Vendor SDKs that build and send their own HTTP. These do not look like
#: network calls at all — ``client.aio.models.generate_content(...)`` reads
#: like a method on an object — which is exactly why the API adapter's cloud
#: path sat outside the gateway without anyone noticing. The constructor is
#: the honest place to catch them: holding one of these clients IS holding a
#: way out of the machine.
_VENDOR_CLIENT_ROOTS = {"genai", "openai", "anthropic", "cohere", "mistralai"}
_VENDOR_CLIENT_CALLS = {"Client", "AsyncClient", "AsyncAnthropic", "AsyncOpenAI"}

#: Files that legitimately hold direct HTTP, with the reason. This list only
#: shrinks.
ALLOWED: dict[str, str] = {
    "core/runtime/network_gateway.py": "is the gateway",
    "core/adapters/chrome_cdp_transport.py": (
        "the Chrome DevTools socket, refused unless the host is loopback"
    ),
    "core/embodiment/unity_bridge.py": "ws://localhost:8765, the local avatar renderer",
}


def _scan(paths: list[Path] | None = None) -> dict[str, set[str]]:
    """Every direct-egress call in every shipped package, by file."""
    found: dict[str, set[str]] = {}
    if paths is None:
        paths = [p for top in _PRODUCTION for p in (PROJECT_ROOT / top).rglob("*.py")]
    for path in paths:
        if "__pycache__" in str(path):
            continue
        try:
            tree = ast.parse(path.read_text())
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        try:
            rel = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            rel = str(path)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue

            # aiohttp.ClientSession(...) / urllib.request.urlopen(...)
            if func.attr in _DIRECT_EGRESS_CALLS:
                found.setdefault(rel, set()).add(func.attr)
                continue

            root = func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if not isinstance(root, ast.Name):
                continue

            # httpx.post(...) / requests.get(...) — a verb on a client root.
            if func.attr in _HTTP_VERBS and root.id in _DIRECT_EGRESS_ROOTS:
                found.setdefault(rel, set()).add(f"{root.id}.{func.attr}")

            # genai.Client(...) — a vendor SDK that carries its own transport.
            elif func.attr in _VENDOR_CLIENT_CALLS and root.id in _VENDOR_CLIENT_ROOTS:
                found.setdefault(rel, set()).add(f"{root.id}.{func.attr}")

            # httpx.AsyncClient(...) / websockets.connect(...) — a way out held
            # in a variable, used by a verb this scan cannot see.
            elif func.attr in _CLIENT_CONSTRUCTORS.get(root.id, ()):
                found.setdefault(rel, set()).add(f"{root.id}.{func.attr}")
    return found


def test_no_new_direct_egress_outside_the_gateway():
    found = _scan()
    offenders = {
        rel: sorted(calls) for rel, calls in found.items() if rel not in ALLOWED
    }
    assert not offenders, (
        "outbound HTTP that skips core/runtime/network_gateway.py — and "
        "therefore skips governance, the outbound preflight, and the egress "
        f"privacy boundary: {offenders}. Route it through "
        "get_network_gateway().request_async()."
    )


def test_the_allowlist_only_shrinks():
    """An allowlist entry that no longer needs to be there is debt, not policy."""
    found = _scan()
    stale = sorted(set(ALLOWED) - set(found))
    assert not stale, (
        f"these files no longer contain direct egress and should be removed "
        f"from ALLOWED: {stale}"
    )


def test_the_scan_sees_a_client_held_in_a_variable(tmp_path):
    """The shapes the scan was blind to, each written the way a person would."""
    samples = {
        "httpx_client.py": "import httpx\nasync def f(b):\n    c = httpx.AsyncClient()\n    await c.post('https://x', json=b)\n",
        "http_client.py": "import http.client\ndef f():\n    http.client.HTTPSConnection('x').request('POST', '/')\n",
        "ws.py": "import websockets\nasync def f():\n    await websockets.connect('wss://x')\n",
        "session.py": "import requests\ndef f():\n    requests.Session().post('https://x')\n",
    }
    paths = []
    for name, body in samples.items():
        target = tmp_path / name
        target.write_text(body)
        paths.append(target)
    found = _scan(paths)
    assert {Path(k).name for k in found} == set(samples), found
