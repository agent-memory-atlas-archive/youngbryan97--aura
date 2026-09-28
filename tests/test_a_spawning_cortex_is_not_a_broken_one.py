"""A model that is loading is not a model that has failed.

LIVE 2026-09-28 01:11, at the moment a personality test asked her a question
about herself. The decision routed to her own lane, the Cortex worker began
spawning — 20GB, seconds to load — and five readiness probes in those seconds
each read `worker_not_alive,init_not_complete,lane_handshaking`. The breaker
counted them: "Circuit OPEN for Cortex after 5 failures", then "no endpoints
matched routing plan for tier 'primary'", then an empty decision, and the run
ended with the test half answered.

`lane_handshaking` and `lane_spawning` are the states a spawn spends its seconds
in, and neither was on the list of reasons that mean "becoming ready".
"""
from __future__ import annotations

import pytest

from core.brain.llm_health_router import (
    _is_transient_local_runtime_failure,
    _only_still_coming_up,
)

pytestmark = pytest.mark.unit


def test_the_live_reason_is_read_as_a_spawn():
    assert _only_still_coming_up("worker_not_alive,init_not_complete,lane_handshaking")
    assert _is_transient_local_runtime_failure(
        "worker_not_alive,init_not_complete,lane_handshaking"
    )


def test_every_arriving_lane_state_counts_as_arriving():
    for state in ("spawning", "handshaking", "warming", "recovering"):
        assert _only_still_coming_up(f"worker_not_alive,lane_{state}"), state


def test_a_worker_that_is_simply_not_there_still_counts():
    """Alone, the reason means a worker that died or was never started."""
    assert not _only_still_coming_up("worker_not_alive")
    assert not _only_still_coming_up("lane_cold,worker_not_alive")


def test_a_real_fault_beside_a_spawn_is_still_a_real_fault():
    assert not _only_still_coming_up("worker_not_alive,runtime_shutdown")
    assert not _only_still_coming_up("worker_not_alive,lane_failed")
    assert not _only_still_coming_up("lane_handshaking,worker_progress_stale")


def test_the_reasons_already_covered_still_are():
    assert _only_still_coming_up("warmup_in_flight,warmup_foreground_owner")
    assert _only_still_coming_up("init_not_complete")
    assert _only_still_coming_up("lane_recovering")


def test_nothing_said_is_not_a_reason_to_forgive():
    assert not _only_still_coming_up("")
    assert not _only_still_coming_up("   ")


def test_every_lane_state_the_client_can_set_is_classified():
    """A new lane state must be decided on, not left to fall through as a fault."""
    import re

    from core.brain.llm_health_router import _LANE_IS_ARRIVING, _STILL_COMING_UP

    states = set()
    for path in (
        "core/brain/llm/mlx_client.py",
        "core/brain/llm/mlx_warmup_and_adapters.py",
        "core/brain/llm/mlx_client_waiting.py",
        "core/brain/llm/mlx_client_worker_lifecycle.py",
    ):
        try:
            source = open(path).read()
        except FileNotFoundError:
            continue
        states.update(re.findall(r'_set_lane_state\(\s*"([a-z_]+)"', source))
        # `_set_lane_state("closed" if proven else "shutdown_failed", ...)`
        states.update(
            re.findall(
                r'_set_lane_state\(\s*"[a-z_]+"\s+if\s+[^,]+?\s+else\s+"([a-z_]+)"',
                source,
            )
        )
    arriving = {"spawning", "handshaking", "warming", "recovering"}
    settled = {"ready", "cold", "failed", "fenced", "closed", "shutdown_failed"}
    unknown = states - arriving - settled
    assert not unknown, (
        f"lane state(s) {sorted(unknown)} are neither arriving nor settled; "
        "decide which before a probe reads one as a failure"
    )
    for state in arriving:
        assert f"lane_{state}" in _LANE_IS_ARRIVING
        assert f"lane_{state}" in _STILL_COMING_UP or state in {"warming", "recovering"}
