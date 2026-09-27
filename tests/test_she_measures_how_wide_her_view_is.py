"""She measures her own field of view by following something round the room.

A slide is in pixels and a bearing is in degrees, and nothing in a frame says
how many degrees a pixel is worth. Without that, aiming at a moving thing is
a guess in the wrong units: measured 2026-09-26, leading by a drift in pixels
aimed past the edge of the view and chasing fell from 20 of 30 to 11.

The frames cannot answer it on their own. Turning in whole mouse points does
not land back on the starting heading — 18.09 degrees a step here, so a
revolution takes 19.9 of them and the view never repeats. A landmark crossing
the middle of the view is continuous and can be interpolated, which is what
makes the count exact.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

pytestmark = pytest.mark.unit


def _a_world(seed: int):
    from measure_in_camera_worlds import AGeneratedWorld

    world = AGeneratedWorld(seed)
    world.place("door", 20.0, 8.0, "e")
    return world


async def _measured(seed: int):
    from core.skills.in_a_world_through_a_camera import ACameraWorld, learn_the_body, measure_view

    world = _a_world(seed)
    loop = ACameraWorld(look=world.look, play=world.play)
    body = await learn_the_body(loop, keys=("w", "s"), slot_s=0.2, measure_the_view=False)
    return await measure_view(loop, body, slot_s=0.2)


@pytest.mark.parametrize("seed", (0, 1, 2))
def test_the_field_of_view_is_measured_within_a_few_percent(seed: int) -> None:
    from measure_in_camera_worlds import FIELD

    wide = asyncio.run(_measured(seed))
    assert wide.known, "a room with something named in it told her nothing"
    off_by = abs(wide.degrees_across - FIELD) / FIELD
    assert off_by < 0.05, (
        f"read the view as {wide.degrees_across:.1f} degrees against {FIELD}"
    )


def test_degrees_and_pixels_convert_both_ways() -> None:
    wide = asyncio.run(_measured(0))
    assert wide.known
    assert wide.pixels_for(wide.degrees_for(40.0)) == pytest.approx(40.0, rel=1e-6)
    # A whole view's width is the field of view, by construction.
    across = wide.pixels_for(wide.degrees_across)
    assert wide.degrees_for(across) == pytest.approx(wide.degrees_across, rel=1e-6)


def test_body_learning_wires_the_view_measurement() -> None:
    from core.skills.in_a_world_through_a_camera import ACameraWorld, learn_the_body

    world = _a_world(0)
    loop = ACameraWorld(look=world.look, play=world.play)
    body = asyncio.run(learn_the_body(loop, keys=("w", "s"), slot_s=0.2))
    assert body.view.known


def test_a_world_with_nothing_named_reports_nothing() -> None:
    """No landmark, no measurement — and no guess."""
    from core.skills.in_a_world_through_a_camera import ACameraWorld, learn_the_body, measure_view

    async def run():
        world = _a_world(0)
        world.things.clear()
        loop = ACameraWorld(look=world.look, play=world.play)
        body = await learn_the_body(loop, keys=("w", "s"), slot_s=0.2, measure_the_view=False)
        return await measure_view(loop, body, slot_s=0.2, most_turns=30)

    wide = asyncio.run(run())
    assert not wide.known
    assert wide.degrees_across == 0.0
    assert wide.degrees_for(100.0) == 0.0
    assert wide.pixels_for(90.0) == 0.0


def test_the_lead_never_aims_past_the_thing_she_is_following() -> None:
    """Aiming at the edge of the VIEW is what lost it."""
    from core.agency.going_to_what_she_sees import GoingTo, Sighting

    going = GoingTo("door", turn_for=lambda _share: 0, walks="w", leads=True)
    going._drifts.extend([0.06, 0.06, 0.06])
    going.growths.append(1.02)  # fifty steps out, so an unbounded lead is huge
    seen = Sighting("Door", across=0.10, down=0.0, wide=0.08, high=0.2)
    aim = going._where_it_will_be(seen)
    assert aim > seen.across, "a thing seen moving is led"
    assert aim - seen.across <= seen.wide / 2.0 + 1e-9, "the lead passed the thing's own edge"
    assert abs(aim) < 0.5, "the aim left the view"
