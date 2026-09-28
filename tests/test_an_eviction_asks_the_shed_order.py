"""Memory eviction frees memory, or it is only an announcement.

LIVE 2026-09-28 00:43, while a page run held the foreground at 92% resource
pressure: "ResourceGovernor: Eviction tier=moderate, callbacks=0", then
"tier=soft, callbacks=0". ``register_eviction_callback`` had no caller anywhere
in the runtime, so the governor's whole eviction path ran an empty list and
logged a warning about it. The organs that can actually free memory are
registered with the OOM policy by the container, and nothing asked them.
"""
from __future__ import annotations

import pytest

from core.resource.resource_governor import EvictionTier, ResourceGovernor
from core.runtime.oom_policy import register_organ, reset_oom_policy_for_test

pytestmark = pytest.mark.unit


class _Observer:
    """A host with no room left, so the shed target cannot already be met."""

    def __init__(self) -> None:
        self.total = 64 * 2**30
        self.available = 1 * 2**30

    def memory(self, **_kw):
        import types

        return types.SimpleNamespace(
            total_bytes=self.total, available_bytes=self.available
        )


@pytest.fixture
def clean_policy():
    reset_oom_policy_for_test()
    yield
    reset_oom_policy_for_test()


def test_an_eviction_sheds_a_registered_organ(clean_policy):
    shed_calls: list[int] = []

    register_organ(
        "a_cache",
        oom_score_adj=500,
        footprint=lambda: 2 * 2**30,
        shed=lambda: (shed_calls.append(1), 2 * 2**30)[1],
        rationale="a test cache",
    )
    governor = ResourceGovernor(observer=_Observer())
    governor.execute_eviction(EvictionTier.MODERATE)
    assert shed_calls, "the eviction did not ask the shed order for anything"


def test_the_target_is_the_line_the_tier_fired_at(clean_policy):
    """Each tier sheds back under its own threshold, not a number chosen here."""
    governor = ResourceGovernor(observer=_Observer())
    assert governor._threshold_for(EvictionTier.SOFT) == governor.MEMORY_SOFT_THRESHOLD
    assert (
        governor._threshold_for(EvictionTier.MODERATE)
        == governor.MEMORY_MODERATE_THRESHOLD
    )
    assert (
        governor._threshold_for(EvictionTier.AGGRESSIVE)
        == governor.MEMORY_AGGRESSIVE_THRESHOLD
    )


def test_a_host_with_room_sheds_nothing(clean_policy):
    """A tier can fire on one reading and clear on the next; shedding then is
    taking memory from her for a condition that has passed."""
    shed_calls: list[int] = []
    register_organ(
        "a_cache",
        oom_score_adj=500,
        footprint=lambda: 2 * 2**30,
        shed=lambda: (shed_calls.append(1), 0)[1],
        rationale="a test cache",
    )
    observer = _Observer()
    observer.available = 40 * 2**30
    governor = ResourceGovernor(observer=observer)
    governor.execute_eviction(EvictionTier.MODERATE)
    assert not shed_calls


def test_no_pressure_asks_for_nothing(clean_policy):
    shed_calls: list[int] = []
    register_organ(
        "a_cache", oom_score_adj=500, footprint=lambda: 2 * 2**30,
        shed=lambda: (shed_calls.append(1), 0)[1], rationale="a test cache",
    )
    governor = ResourceGovernor(observer=_Observer())
    assert governor.execute_eviction(EvictionTier.NONE) == 0
    assert not shed_calls
