"""A,S -> G had no product, so the synergy line could not find one.

The workspace's priority is a sum: base salience, what affect lends, the focus
bias, a free-energy bias, held pressure, civility debt, less what saying it
already drained. Affect lent three tenths of its weight *whatever the self-model
said*, so on the whole seed-7 run of 27 September the A,S -> G interaction gain
came out at exactly +0.00000 — the estimator finding no product because there was
none to find.

The claim is about minds rather than about the score: a feeling is a guide to what
matters only in so far as she can read her own state. Her self-prediction
publishes exactly that confidence, so the lend is scaled by it.
"""

from __future__ import annotations

import pytest

from core.consciousness import global_workspace as gw


class _Prediction:
    def __init__(self, confidence) -> None:
        self.confidence = confidence


class _Loop:
    def __init__(self, prediction) -> None:
        self._current_prediction = prediction


@pytest.fixture
def self_prediction(monkeypatch):
    def _set(prediction):
        from core.runtime import service_registry

        monkeypatch.setattr(
            service_registry,
            "get_runtime_service",
            lambda name, default=None: _Loop(prediction) if name == "self_prediction" else default,
        )

    return _set


# ── the reading ──────────────────────────────────────────────────────────


def test_no_self_prediction_lends_in_full(monkeypatch):
    """The behaviour before this existed."""
    from core.runtime import service_registry

    monkeypatch.setattr(
        service_registry, "get_runtime_service", lambda name, default=None: default
    )
    assert gw._reading_herself() == 1.0


def test_a_loop_with_no_prediction_yet_lends_in_full(self_prediction):
    self_prediction(None)
    assert gw._reading_herself() == 1.0


def test_the_confidence_is_the_organ_s_own(self_prediction):
    self_prediction(_Prediction(0.4))
    assert gw._reading_herself() == pytest.approx(0.4)


@pytest.mark.parametrize("value", [-1.0, 2.0, float("nan"), "sure", None])
def test_an_unusable_confidence_lends_in_full_or_stays_in_range(self_prediction, value):
    self_prediction(_Prediction(value))
    reading = gw._reading_herself()
    assert 0.0 <= reading <= 1.0


# ── the product it makes ─────────────────────────────────────────────────


def _bid(**kw):
    from core.consciousness.global_workspace import ContentType

    defaults = {
        "source": "percept",
        "content": {},
        "priority": 0.2,
        "content_type": ContentType.PERCEPTUAL,
        "affect_weight": 1.0,
    }
    defaults.update(kw)
    return gw.CognitiveCandidate(**defaults)


def test_affect_lends_less_when_she_is_reading_herself_less_well(self_prediction):
    bid = _bid()
    self_prediction(_Prediction(1.0))
    sure = bid.priority_at(bid.submitted_at)
    self_prediction(_Prediction(0.1))
    unsure = bid.priority_at(bid.submitted_at)
    assert sure > unsure, "the lend has to depend on the self-model or there is no product"


def test_the_two_together_are_not_the_sum_of_the_two_apart(self_prediction):
    """What an interaction means: the effect of one depends on the other."""
    quiet = _bid(affect_weight=0.0)
    loud = _bid(affect_weight=1.0)

    self_prediction(_Prediction(1.0))
    quiet_sure = quiet.priority_at(quiet.submitted_at)
    loud_sure = loud.priority_at(loud.submitted_at)
    self_prediction(_Prediction(0.2))
    quiet_unsure = quiet.priority_at(quiet.submitted_at)
    loud_unsure = loud.priority_at(loud.submitted_at)

    # Affect's effect differs by how well she reads herself; the self-model's
    # effect differs by how much affect there is. Either statement is the same
    # product, and a sum would make both differences zero.
    assert (loud_sure - quiet_sure) > (loud_unsure - quiet_unsure)
    assert (loud_sure - loud_unsure) > (quiet_sure - quiet_unsure)


def test_a_bid_that_is_itself_the_feeling_still_lends_nothing(self_prediction):
    from core.consciousness.global_workspace import ContentType

    self_prediction(_Prediction(1.0))
    affective = _bid(content_type=ContentType.AFFECTIVE, affect_weight=1.0)
    bare = _bid(content_type=ContentType.AFFECTIVE, affect_weight=0.0)
    assert affective.priority_at(affective.submitted_at) == pytest.approx(
        bare.priority_at(bare.submitted_at)
    )


def test_the_three_tenths_is_unchanged(self_prediction):
    """Only the factor is new; the weight the workspace already declared stays."""
    import inspect

    source = inspect.getsource(gw.CognitiveCandidate.priority_at)
    assert "self.affect_weight * 0.3 * _reading_herself()" in source
