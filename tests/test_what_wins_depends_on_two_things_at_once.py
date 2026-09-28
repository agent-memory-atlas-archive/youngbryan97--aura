"""What takes the workspace depends on her feeling and her self-state together.

`_bids` summed what each bid was made of and weighed each part, so two domains
feeding one competition added and never interacted. On whole-s7-27dc1dda9 the
held-out interaction gain of A and S about G was exactly 0.0 with a lower bound
of -0.00851, which is what a sum looks like to an estimator asking for a product.

Divisive normalisation makes it a product: each bid is divided by the pool it
sits in, in proportion to how little she can read herself.
"""

from __future__ import annotations

import pytest

from core.consciousness.global_workspace import _normalised_by_the_pool


def test_a_field_of_one_is_never_touched() -> None:
    assert _normalised_by_the_pool({1: 0.7}) == {1: 0.7}


def test_sure_of_herself_the_scores_stand(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.consciousness.global_workspace._reading_herself", lambda: 1.0)
    raw = {1: 0.9, 2: 0.3, 3: 0.1}
    assert _normalised_by_the_pool(raw) == raw


def test_unsure_of_herself_the_pool_compresses_the_loud_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.consciousness.global_workspace._reading_herself", lambda: 0.0)
    raw = {1: 0.9, 2: 0.3, 3: 0.1}
    out = _normalised_by_the_pool(raw)
    assert out[1] < raw[1], "the loudest bid was not compressed"
    spread_before = max(raw.values()) - min(raw.values())
    spread_after = max(out.values()) - min(out.values())
    assert spread_after < spread_before
    # And the order is kept: compression is not a reshuffle.
    assert sorted(out, key=lambda k: out[k]) == sorted(raw, key=lambda k: raw[k])


def test_the_gap_between_two_bids_depends_on_both_her_feeling_and_her_self_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The interaction, stated as the thing the criterion measures.

    A feeling raises one bid. How much of that reaches the decision depends on
    how well she reads herself, so the effect of the feeling is not the same at
    both self-states, which is exactly a non-zero second difference.
    """

    def gap(feeling: float, herself: float) -> float:
        monkeypatch.setattr(
            "core.consciousness.global_workspace._reading_herself", lambda: herself
        )
        out = _normalised_by_the_pool({1: 0.4 + feeling, 2: 0.4})
        return out[1] - out[2]

    low_sure, high_sure = gap(0.0, 1.0), gap(0.5, 1.0)
    low_unsure, high_unsure = gap(0.0, 0.0), gap(0.5, 0.0)
    effect_when_sure = high_sure - low_sure
    effect_when_unsure = high_unsure - low_unsure
    assert effect_when_sure > effect_when_unsure, (effect_when_sure, effect_when_unsure)
    assert abs(effect_when_sure - effect_when_unsure) > 0.01


def test_a_pool_that_is_all_zero_is_returned_as_it_came(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("core.consciousness.global_workspace._reading_herself", lambda: 0.0)
    raw = {1: 0.0, 2: -0.2}
    assert _normalised_by_the_pool(raw) == raw
