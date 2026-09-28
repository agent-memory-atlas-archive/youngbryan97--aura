"""Two arms from one anchor are the same computation only once the free loops stop.

The harness steps each of her free-running layers once a frame, at the layer's
own rate (core/subject/steppable.py). The v25 cut sweep and the reports ground
forked their arms with each layer's own loop still running as well, ticking it
on the machine's clock on top of the harness's step, so two untouched forks
from one anchor parted within a frame or three. On seed 7 on 28 September that
was 65 to 109 of 438 columns apart by the end of a turn, and the sham floor of
the v25 look at 38ef2c9ce read 0.062 to 0.093, more than seven singleton cuts'
whole effect. run_subject_core.py already stopped them before its
interventions; both runners now do too.

A run with her cortex stops only the stepped loops, because her language organ
keeps tasks of its own and needs them to answer.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from core.subject.organism import quiesce, stepped_loops
from core.subject.steppable import LAYERS


def test_every_stepped_layer_is_a_loop_that_stops() -> None:
    assert set(stepped_loops()) == {layer.name for layer in LAYERS}


def _tasks_left_after(only: tuple[str, ...] | None) -> set[str]:
    async def run() -> set[str]:
        async def forever() -> None:
            await asyncio.Event().wait()

        names = (
            "NeuralMesh",
            "StreamOfBeing.existence",
            "ClosedCausalLoop.prediction",
            "InferenceGate.maintenance",
            "NeuralMeshology",
        )
        tasks = [asyncio.create_task(forever(), name=name) for name in names]
        await asyncio.sleep(0)
        await quiesce(only=only)
        alive = {task.get_name() for task in tasks if not task.done()}
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return alive

    return asyncio.run(run())


def test_the_stepped_loops_stop_and_her_cortex_keeps_its_own() -> None:
    alive = _tasks_left_after(stepped_loops())
    assert alive == {"InferenceGate.maintenance", "NeuralMeshology"}


def test_with_no_list_every_loop_stops() -> None:
    assert _tasks_left_after(None) == set()


@pytest.mark.slow
def test_two_forks_agree_in_the_layers_the_loops_drove() -> None:
    """Fork one anchor twice after the stepped loops stop, and read the layers they ticked."""
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )
    from core.subject.state import feature_names

    driven = ("C.mesh_", "C.field_")

    async def run() -> tuple[np.ndarray, np.ndarray, list[str], set[str]]:
        with TemporaryDirectory() as tmp:
            # Its own state root, as every runner takes one.
            from core.subject.isolation import isolate_state

            isolate_state(Path(tmp))
            runtime = build_runtime(Path(tmp) / "runtime", seed=7)
            await start_organism(runtime)
            await calibrate_clock(runtime, CONDITIONS, turns=1)
            for condition in CONDITIONS[:2]:
                await runtime.turn_once(condition)
            await quiesce_organism(runtime, stepped_only=True)
            runtime.freeze_host()
            snapshot = runtime.snapshot()
            memory = next(c for c in CONDITIONS if c.name == "memory")
            ends = []
            for _ in range(2):
                runtime.restore(snapshot)
                frames = await runtime.turn_once(memory)
                ends.append(np.asarray(frames[-1].vector(), dtype=np.float64))
            live = {task.get_name() for task in asyncio.all_tasks() if not task.done()}
            await quiesce_organism(runtime)
            return ends[0], ends[1], list(feature_names()), live

    first, second, names, live = asyncio.run(run())
    assert not {name for name in live if name in stepped_loops()}, f"a stepped loop still runs: {live}"
    columns = [i for i, name in enumerate(names) if name.startswith(driven)]
    assert columns, "no mesh or field columns in the schema"
    apart = {names[i]: float(abs(first[i] - second[i])) for i in columns if abs(first[i] - second[i]) > 1e-9}
    assert not apart, f"two forks parted in the layers the loops drove: {apart}"
