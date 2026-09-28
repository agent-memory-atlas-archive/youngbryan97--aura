"""A subsystem that moves the sampler must be declared, or it moves nothing.

The gate filters kwargs to the declared request fields and logs the rest at
debug level. A bias published by a subsystem and absent from that schema is
computed on every turn and dropped in silence.

LIVE, 2026-08-28: the cognitive-situation frame was the fourth such channel and
was in neither the gate's list of bias keys nor the typed request schema. Its
own test proved the engine handed the bias to a stub router and stopped there,
which is the shape of a half-wired channel: a writer, a test of the writer, and
no reader.

This is the check that would have caught it. The channels are read off the
request the desktop path actually hands the router, rather than from a list,
so a fifth channel added tomorrow cannot go quiet.
"""

from __future__ import annotations

import asyncio
from functools import lru_cache

from core.brain.inference_gate import _SAMPLING_BIAS_KEYS
from core.brain.request_contract import REQUEST_FIELDS


@lru_cache(maxsize=1)
def _published_channels() -> frozenset[str]:
    """Every sampling bias the engine hands the router on a desktop turn.

    Taken from one real turn of the quick-reply path against a router that
    keeps what it was given. A channel is a key of that request; its value
    may be empty on this turn and it is still a channel.
    """
    from core.brain.cognitive_engine import CognitiveEngine
    from core.brain.types import ThinkingMode
    from core.container import ServiceContainer

    given: dict = {}

    class _Router:
        async def think(self, messages, **kwargs):
            given.update(kwargs)
            return "I would ground the visible state first, then act through the governed lane."

        def get_last_generation_metadata(self):
            return {}

    ServiceContainer.clear()
    ServiceContainer.register_instance("llm_router", _Router(), required=False)
    try:
        asyncio.run(
            CognitiveEngine()._direct_desktop_quick_reply(
                "Can you compare this task to a checklist?",
                ThinkingMode.FAST,
                "desktop",
                {
                    "desktop_quick_reply_contract": True,
                    "desktop_cognitive_engine_required": True,
                    "max_tokens": 512,
                },
                timeout_s=20.0,
            )
        )
    finally:
        ServiceContainer.clear()
    return frozenset(key for key in given if key.endswith("sampling_bias"))


def test_the_engine_publishes_the_channels_we_think_it_does() -> None:
    published = _published_channels()
    assert len(published) >= 4, published
    assert "cognitive_situation_sampling_bias" in published


def test_the_reader_side_is_not_empty_either() -> None:
    """A blind writer scan and a blind reader scan look identical from here.

    The failure this file once had was a scan that found nothing and reported
    it as a set, so a second empty set could hide the same way.
    """
    assert len(_SAMPLING_BIAS_KEYS) >= 4, _SAMPLING_BIAS_KEYS
    assert set(_SAMPLING_BIAS_KEYS) <= set(REQUEST_FIELDS), sorted(
        set(_SAMPLING_BIAS_KEYS) - set(REQUEST_FIELDS)
    )


def test_a_published_bias_is_declared_in_the_request_schema() -> None:
    for channel in sorted(_published_channels()):
        assert channel in REQUEST_FIELDS, (
            f"{channel} is published by a subsystem and not declared, so the gate "
            "filters it out and the subsystem moves nothing"
        )


def test_a_published_bias_is_read_by_the_gate() -> None:
    for channel in sorted(_published_channels()):
        assert channel in _SAMPLING_BIAS_KEYS, (
            f"{channel} is declared but not among the gate's bias keys"
        )


def test_the_generation_phase_reads_the_same_set() -> None:
    """Three lists of the same thing is how one of them goes stale.

    Each channel alone carries a warming bias into the generation phase's
    sampling step, and each one moves the temperature it settles on.
    """
    from core.phases.response_generation import ResponseGenerationPhase
    from core.state.aura_state import AuraState

    phase = ResponseGenerationPhase.__new__(ResponseGenerationPhase)

    def temperature_with(modifiers: dict) -> float:
        state = AuraState.default()
        state.response_modifiers.update(modifiers)
        temperature, _tokens = phase._execute_affect_modulated_generation(
            False, False, False, None, {}, state
        )
        return temperature

    resting = temperature_with({})
    for channel in sorted(_published_channels()):
        moved = temperature_with({channel: {"temperature_delta": 0.15}})
        assert moved > resting, f"the generation phase does not read {channel}"
