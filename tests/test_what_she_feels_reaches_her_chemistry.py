"""Her appraised emotions release the chemicals that emotion releases.

`EmotionSignatureEngine` holds a neurochemical recipe for eight emotions and
the neurochemical system asks it for a production modifier on every metabolic
tick. Until 28 September `set_emotion` had no caller outside its own module, so
`emotion_intensity` stayed at the 0.0 its constructor set and every recipe
contributed exactly zero to every chemical for the life of the process.

Her chemistry is what the consciousness bridge writes into the substrate's
valence neuron on every frame, and that neuron is what steers her cortex. So
the dead edge meant a feeling she appraised could not reach the words she chose:
on seed 7 a held displacement that moved her computed valence from 0.272 to
0.527 moved the substrate valence the hooks read from 0.111 to 0.117.

Each test here names one link of that chain.
"""

from __future__ import annotations

import pytest

from core.consciousness.emotion_signatures import (
    HER_CHANNELS,
    EmotionSignatureEngine,
    get_emotion_signature_engine,
)


def _mood_after(emotions: dict[str, float], ticks: int) -> dict[str, float]:
    """Her mood vector after this many metabolic ticks feeling this."""
    import core.consciousness.emotion_signatures as signatures
    from core.consciousness.neurochemical_system import NeurochemicalSystem

    signatures._emotion_engine = None
    engine = get_emotion_signature_engine()
    system = NeurochemicalSystem()
    for _ in range(ticks):
        engine.feel(emotions)
        system._metabolic_tick()
    return system.get_mood_vector()


def test_every_recipe_has_a_channel_she_can_feel() -> None:
    """A recipe no channel of hers maps to can never be released."""
    from core.consciousness.emotion_signatures import _EMOTION_SIGNATURES
    from core.phases.affect_update import (
        _NEGATIVE_AFFECT_WEIGHTS,
        _POSITIVE_AFFECT_WEIGHTS,
    )

    hers = set(_POSITIVE_AFFECT_WEIGHTS) | set(_NEGATIVE_AFFECT_WEIGHTS)
    unreachable = [name for name in _EMOTION_SIGNATURES if name not in set(HER_CHANNELS.values())]
    assert unreachable == [], f"recipes nothing she feels can release: {unreachable}"
    unknown = [channel for channel in HER_CHANNELS if channel not in hers]
    assert unknown == [], f"the bridge names channels she has no weight for: {unknown}"


def test_a_feeling_moves_her_chemistry_the_way_it_is_felt() -> None:
    """Good feelings raise the mood her chemistry reports; bad ones lower it."""
    flat = _mood_after({}, 20)
    good = _mood_after({"joy": 0.8, "interest": 0.5, "wonder": 0.4}, 20)
    bad = _mood_after({"sadness": 0.8, "fear": 0.6, "boredom": 0.3}, 20)

    assert good["valence"] > flat["valence"] + 0.1, (good["valence"], flat["valence"])
    assert bad["valence"] < flat["valence"] - 0.1, (bad["valence"], flat["valence"])
    assert bad["stress"] > flat["stress"] + 0.1, (bad["stress"], flat["stress"])


def test_feeling_nothing_leaves_her_chemistry_where_it_was() -> None:
    """The edge is a release, not a tonic driver: an empty reading changes nothing."""
    import core.consciousness.emotion_signatures as signatures
    from core.consciousness.neurochemical_system import NeurochemicalSystem

    signatures._emotion_engine = None
    system = NeurochemicalSystem()
    before = system.get_mood_vector()["valence"]
    engine = get_emotion_signature_engine()
    for _ in range(20):
        engine.feel({})
        system._metabolic_tick()
    assert engine.get_neurochemical_modulation() == pytest.approx(
        dict.fromkeys(engine.get_neurochemical_modulation(), 0.0)
    )
    assert system.get_mood_vector()["valence"] == pytest.approx(before, abs=0.01)


def test_several_feelings_at_once_average_by_how_strongly_each_is_felt() -> None:
    """One hot channel reduces to the single-emotion case the engine already had."""
    one = EmotionSignatureEngine()
    one.feel({"joy": 0.7})
    lone = EmotionSignatureEngine()
    lone.set_emotion("joy", 0.7)
    assert one.get_neurochemical_modulation() == pytest.approx(lone.get_neurochemical_modulation())

    both = EmotionSignatureEngine()
    both.feel({"joy": 0.5, "sadness": 0.5})
    mixed = both.get_neurochemical_modulation()
    sorrow = EmotionSignatureEngine()
    sorrow.feel({"sadness": 0.5})
    joy = EmotionSignatureEngine()
    joy.feel({"joy": 0.5})
    for name, value in mixed.items():
        low, high = sorted((joy.get_neurochemical_modulation()[name], sorrow.get_neurochemical_modulation()[name]))
        assert low - 1e-9 <= value <= high + 1e-9, (name, value, low, high)


def test_a_channel_with_no_recipe_releases_nothing() -> None:
    """Thirty-two of her forty channels have no recipe, and that is not an error."""
    engine = EmotionSignatureEngine()
    assert engine.feel({"belonging": 0.9, "nostalgia": 0.4}) == 0.0
    assert engine.get_neurochemical_modulation() == pytest.approx(
        dict.fromkeys(engine.get_neurochemical_modulation(), 0.0)
    )


def test_the_affect_phase_is_what_tells_her_chemistry() -> None:
    """The dead edge was a method with no caller; this is the caller."""
    import inspect

    from core.phases.affect_update import AffectUpdatePhase

    source = inspect.getsource(AffectUpdatePhase._release_chemistry)
    assert "get_emotion_signature_engine" in source
    assert ".feel(" in source
    executed = inspect.getsource(AffectUpdatePhase.execute)
    assert "_release_chemistry" in executed
