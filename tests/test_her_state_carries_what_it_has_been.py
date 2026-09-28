"""The membrane applied to her state, so her next phase reads the trace too.

`core/runtime/state_membrane.py`. A membrane only the recorder saw would be a
change to the measurement rather than to her, so `settle` writes the trace back
where the reading came from and every later reader sees it. Which channels
carry one is found in her own state, not in the battery's column list: every
float that has moved both ways.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from core.runtime.state_membrane import (
    MembraneScope,
    after_phase,
    her_channels,
    membrane_turns,
    settle,
)


@dataclass
class _Affect:
    valence: float = 0.0
    arousal: float = 0.0
    emotions: dict[str, float] = field(default_factory=lambda: {"joy": 0.2})


@dataclass
class _Cognition:
    depth: float = 0.0
    label: str = "not a number"
    turns: int = 3
    origin_is_user: bool = True
    last_tick: float = 1000.0


@dataclass
class _State:
    affect: _Affect = field(default_factory=_Affect)
    cognition: _Cognition = field(default_factory=_Cognition)


def _scope(turns: float = 1.0, frames: float = 33.0) -> MembraneScope:
    return MembraneScope(frames, turns=turns)


def _moved_both_ways(state: _State, scope: MembraneScope, path: str, low: float, high: float) -> None:
    """Take a channel up and down once, so it carries from here on."""
    owner, name = path.rsplit(".", 1)
    holder = getattr(state, owner)
    for value in (low, high, low):
        setattr(holder, name, value)
        settle(state, scope)


def test_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control arm of the campaign is the staircase she has now."""
    monkeypatch.delenv("AURA_MEMBRANE_TURNS", raising=False)
    assert membrane_turns() == 0.0
    monkeypatch.setenv("AURA_MEMBRANE_TURNS", "1")
    assert membrane_turns() == 1.0
    monkeypatch.setenv("AURA_MEMBRANE_TURNS", "not a number")
    assert membrane_turns() == 0.0


def test_her_channels_are_the_floats_in_her_state() -> None:
    """Ints are counts, booleans are flags and strings are labels; none is a quantity."""
    channels = her_channels(_State())
    assert set(channels) == {
        "affect.valence",
        "affect.arousal",
        "affect.emotions.joy",
        "cognition.depth",
        "cognition.last_tick",
    }


def test_a_scope_that_is_off_leaves_every_channel_alone() -> None:
    state = _State(_Affect(valence=0.9))
    out = settle(state, _scope(turns=0.0))
    assert out == {"on": False, "carried": 0, "skipped": 0}
    assert state.affect.valence == 0.9


def test_a_channel_carries_once_it_has_moved_both_ways() -> None:
    state = _State()
    scope = _scope()
    _moved_both_ways(state, scope, "affect.valence", 0.2, 0.6)
    assert "affect.valence" in scope.channels
    assert state.affect.valence == pytest.approx(0.2)
    state.affect.valence = 1.0
    settle(state, scope)
    assert state.affect.valence == pytest.approx(0.2 * scope.membrane.keep + (1.0 - scope.membrane.keep))


def test_before_it_has_moved_both_ways_a_channel_keeps_what_its_writer_wrote() -> None:
    state = _State()
    scope = _scope()
    settle(state, scope)
    state.affect.arousal = 0.7
    settle(state, scope)
    assert state.affect.arousal == 0.7
    assert "affect.arousal" not in scope.channels


def test_a_clock_is_never_carried() -> None:
    """A value that only grows is a clock or a running total, and a trace of a clock is a late clock."""
    state = _State()
    scope = _scope()
    for tick in range(20):
        state.cognition.last_tick = 1000.0 + tick
        settle(state, scope)
        assert state.cognition.last_tick == 1000.0 + tick
    assert "cognition.last_tick" not in scope.channels


def test_a_value_in_a_map_is_written_back_into_the_map() -> None:
    state = _State()
    scope = _scope()
    for joy in (0.1, 0.5, 0.1):
        state.affect.emotions["joy"] = joy
        settle(state, scope)
    state.affect.emotions["joy"] = 0.9
    settle(state, scope)
    assert 0.1 < state.affect.emotions["joy"] < 0.9


def test_forgetting_leaves_a_fork_without_a_life_it_did_not_live() -> None:
    state = _State()
    scope = _scope()
    _moved_both_ways(state, scope, "affect.valence", 0.2, 0.6)
    assert len(scope.membrane) > 0
    scope.forget()
    assert len(scope.membrane) == 0


def test_a_deep_copy_of_her_state_carries_the_traces_and_the_record_of_movement() -> None:
    import copy

    state = _State()
    scope = _scope()
    _moved_both_ways(state, scope, "affect.valence", 0.2, 0.6)
    state.membrane = scope  # type: ignore[attr-defined]
    forked = copy.deepcopy(state)
    assert forked.membrane.membrane.trace("affect.valence") == pytest.approx(0.2)
    assert forked.membrane.channels == scope.channels


def test_after_a_phase_it_settles_only_when_switched_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURA_MEMBRANE_TURNS", raising=False)
    state = _State()
    assert after_phase(state, 29) is state
    assert not hasattr(state, "membrane")
    monkeypatch.setenv("AURA_MEMBRANE_TURNS", "1")
    for value in (0.2, 0.6, 0.2, 1.0):
        state.affect.valence = value
        after_phase(state, 29)
    assert isinstance(state.membrane, MembraneScope)  # type: ignore[attr-defined]
    assert state.membrane.frames_per_turn == 29.0  # type: ignore[attr-defined]
    assert 0.2 < state.affect.valence < 1.0


def test_her_kernel_settles_it_after_each_phase(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hers wherever she runs, not a mechanism only the harness has."""
    from types import SimpleNamespace

    from core.kernel import aura_kernel
    from core.kernel.aura_kernel import AuraKernel

    settled: list[float] = []
    monkeypatch.setattr(aura_kernel, "after_phase", lambda state, frames: settled.append(frames) or state)

    class _Phase:
        async def execute(self, state, **_kwargs):
            return state

    kernel = object.__new__(AuraKernel)
    kernel.state = _State()
    kernel._phases = [_Phase(), _Phase(), _Phase()]
    kernel._phase_runtime_samples = []
    entry = SimpleNamespace(phase_durations_ms={})
    asyncio.run(
        kernel._execute_phase_with_timing(kernel._phases[0], "_Phase", entry, objective="a turn", priority=False)
    )
    assert settled == [3]


def test_the_driver_settles_before_the_clamp_and_before_it_reads() -> None:
    """A trace of a held channel would move the side a lesion is holding."""
    import inspect

    from core.subject.driver import SubjectRuntime

    source = inspect.getsource(SubjectRuntime.turn_once)
    settle_at = source.index("_settle_membrane()")
    clamp_at = source.index("if self.after_phase is not None:")
    read_at = source.index("reading = self.read(")
    assert settle_at < clamp_at < read_at


def test_settling_is_not_a_turn_and_takes_no_condition() -> None:
    """It sat under `turn_once`'s decorator and took the turn's argument.

    `@opens_the_turn` wraps what follows it, so a method written directly above
    `async def turn_once` is the method the decorator lands on. Every frame of
    every turn then raised `missing 1 required positional argument: condition`
    before a single channel carried anything.
    """
    import inspect

    from core.subject.driver import SubjectRuntime

    signature = inspect.signature(SubjectRuntime._settle_membrane)
    assert list(signature.parameters) == ["self"]
    assert not hasattr(SubjectRuntime._settle_membrane, "__wrapped__")


def test_the_scope_rides_on_her_state_so_a_fork_carries_it() -> None:
    """It lived on the runtime, which `restore` does not touch.

    So the second arm of a paired trial began with the first arm's traces: a
    sham arm that was not the same arm. Her state is deep-copied from the
    snapshot, so on the state both arms begin from the anchor's own history.
    """
    import inspect

    from core.subject.driver import SubjectRuntime

    source = inspect.getsource(SubjectRuntime._settle_membrane)
    assert 'getattr(self.state, "membrane"' in source
    assert "self.state.membrane = scope" in source
    assert "self._membrane" not in source
