"""The membrane applied to her state, so her next phase reads the trace too.

`core/runtime/state_membrane.py`. A membrane only the recorder saw would be a
change to the measurement rather than to her, so `settle` writes the trace back
where the reading came from and every later reader sees it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from core.runtime.state_membrane import (
    MembraneScope,
    eligible_channels,
    membrane_turns,
    settle,
)


@dataclass
class _Affect:
    valence: float = 0.0
    arousal: float = 0.0


@dataclass
class _Cognition:
    depth: float = 0.0
    label: str = "not a number"


@dataclass
class _State:
    affect: _Affect
    cognition: _Cognition


class _Scoped(MembraneScope):
    """A scope over four named channels, so a test does not ride on the schema."""

    @property
    def channels(self) -> dict[str, str]:
        return {
            "A.valence": "affect.valence",
            "C.depth": "cognition.depth",
            "C.label": "cognition.label",
            "D.nowhere": "motivation.absent",
        }


def _scope(turns: float = 1.0, frames: float = 33.0) -> MembraneScope:
    return _Scoped(frames, turns=turns)


def test_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control arm of the campaign is the staircase she has now."""
    monkeypatch.delenv("AURA_MEMBRANE_TURNS", raising=False)
    assert membrane_turns() == 0.0
    monkeypatch.setenv("AURA_MEMBRANE_TURNS", "1")
    assert membrane_turns() == 1.0
    monkeypatch.setenv("AURA_MEMBRANE_TURNS", "not a number")
    assert membrane_turns() == 0.0


def test_a_scope_that_is_off_leaves_every_channel_alone() -> None:
    state = _State(_Affect(valence=0.9), _Cognition(depth=0.5))
    out = settle(state, _scope(turns=0.0))
    assert out == {"on": False, "carried": 0, "skipped": 0}
    assert state.affect.valence == 0.9


def test_the_trace_is_written_back_where_the_reading_came_from() -> None:
    """So her own next phase reads it, not only whatever records her."""
    state = _State(_Affect(valence=0.0), _Cognition(depth=0.0))
    scope = _scope()
    settle(state, scope)  # the first frame takes the value outright
    assert state.affect.valence == pytest.approx(0.0)
    state.affect.valence = 1.0
    settle(state, scope)
    assert 0.0 < state.affect.valence < 1.0
    assert state.affect.valence == pytest.approx(1.0 - scope.membrane.keep, abs=1e-9)


def test_a_channel_that_is_not_a_number_is_left_to_its_owner() -> None:
    state = _State(_Affect(), _Cognition(depth=0.2, label="still a string"))
    out = settle(state, _scope())
    assert state.cognition.label == "still a string"
    assert out["skipped"] >= 1


def test_a_boolean_is_not_carried() -> None:
    """A flag is what it is; a trace of it is a number nothing declared."""

    @dataclass
    class _Flagged:
        origin_is_user: bool = True

    state = _State(_Affect(), _Cognition())
    state.cognition = _Flagged()  # type: ignore[assignment]
    class _OneFlag(MembraneScope):
        @property
        def channels(self) -> dict[str, str]:
            return {"D.origin_is_user": "cognition.origin_is_user"}

    settle(state, _OneFlag(33.0, turns=1.0))
    assert state.cognition.origin_is_user is True


def test_a_path_that_is_not_there_is_counted_and_not_raised() -> None:
    state = _State(_Affect(), _Cognition())
    out = settle(state, _scope())
    assert out["skipped"] >= 1
    assert out["carried"] >= 1


def test_forgetting_leaves_a_fork_without_a_life_it_did_not_live() -> None:
    state = _State(_Affect(valence=0.4), _Cognition(depth=0.1))
    scope = _scope()
    settle(state, scope)
    assert len(scope.membrane) > 0
    scope.forget()
    assert len(scope.membrane) == 0


def test_every_eligible_channel_is_a_plain_path_on_her_state() -> None:
    """An organ reading and a dug-out list are not fields anything can write back."""
    channels = eligible_channels()
    assert len(channels) > 150
    for column, path in channels.items():
        assert not path.startswith("organ:"), column
        assert "[" not in path and "*" not in path, column
        assert "." in column


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


def test_a_deep_copy_of_her_state_carries_the_traces_and_not_the_table() -> None:
    import copy

    state = _State(_Affect(valence=0.5), _Cognition(depth=0.2))
    scope = _scope()
    settle(state, scope)
    state.membrane = scope  # type: ignore[attr-defined]
    forked = copy.deepcopy(state)
    assert forked.membrane.membrane.trace("A.valence") == pytest.approx(0.5)
    # The channel table is shared rather than copied per arm.
    assert forked.membrane.channels == scope.channels
