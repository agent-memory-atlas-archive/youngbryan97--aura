"""The turn in front of her is waiting on it, so it is not background work.

LIVE 2026-09-28: every self-report answer came back
`not_her_own_reasoning:suppressed`, and the forecast of the result was never
said at all. The response phase classifies work by the origin's NAME, and a
cycle run from inside a tool the person asked for does not have a user-facing
name — so a question she was told to answer was stood down as background while
the very turn that asked it waited.

The claim is not taken on its own. A flag saying "this is foreground" would be
an unauthenticated claim on the foreground lane, so it counts only alongside the
caller's declared requirement that the answer come from her own lane.
"""
from __future__ import annotations

import inspect

import pytest

from core.phases import response_generation
from core.skills import sovereign_browser_understanding as u  # noqa: F401

pytestmark = pytest.mark.unit


def _classification() -> str:
    body = inspect.getsource(response_generation.ResponseGenerationPhase.execute)
    return body.split("is_background = ", 1)[1].split("foreground_user_surface_owned", 1)[0]


def test_serving_the_current_turn_is_not_background():
    assert "serves_this_turn" in _classification()


def test_the_claim_needs_the_declared_requirement_beside_it():
    body = inspect.getsource(response_generation.ResponseGenerationPhase.execute)
    claim = body.split("serves_this_turn = ", 1)[1].split("is_background = ", 1)[0]
    assert "serves_current_turn" in claim
    assert "own_lane_required" in claim, (
        "an unpaired foreground claim is a claim anything could make"
    )


def test_an_ordinary_background_origin_is_still_background():
    body = inspect.getsource(response_generation.ResponseGenerationPhase.execute)
    assert "background_policy.is_user_facing_origin(origin)" in body, (
        "the origin rule must still decide everything that makes no claim"
    )

def test_the_browser_makes_both_claims_on_its_own_lane_call():
    """The page work that the turn is waiting on says so where it asks."""
    body = inspect.getsource(u._UnderstandsThePage._asked_of_her)
    assert "own_lane_required=True" in body
    assert "serves_current_turn=True" in body
