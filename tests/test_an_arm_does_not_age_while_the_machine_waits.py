"""An arm reads the same however long the machine waited before running it.

A subject run rewinds `time.time` with every restore. A phase report the
binding stamped on the monotonic clock did not rewind: restored into an arm, it
was as old as the real time since the snapshot, and past two seconds the
binding's synchrony fell to its neutral 0.5. Any arm run later in a run read
the field differently from the same arm run earlier. On 30 September that was
a memory arm reading 0.000231 differently after a tool-use arm than after a
conversation arm, and 5 seconds of waiting reproduced it in every condition.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

#: The arm-order test's tolerance.
TOLERANCE = 1e-6


def test_a_phase_report_ages_on_the_clock_a_run_rewinds(monkeypatch) -> None:
    from core.consciousness.oscillatory_binding import OscillatoryBinding

    now = [1_000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    binding = OscillatoryBinding()
    for source in ("mesh", "substrate", "chemicals"):
        binding.report_phase(source, 1.0)
    binding._compute_synchronization()
    assert binding.get_psi() > 0.9

    now[0] += 3.0
    binding._compute_synchronization()
    assert binding.get_psi() == pytest.approx(0.5)


@pytest.mark.slow
def test_an_arm_reads_the_same_after_the_machine_waited(monkeypatch, tmp_path: Path) -> None:
    from core.config import config
    from core.subject.clock import installed_clock
    from core.subject.driver import (
        CONDITIONS,
        build_runtime,
        calibrate_clock,
        quiesce_organism,
        start_organism,
    )

    state = tmp_path / "state"
    state.mkdir()
    monkeypatch.setenv("AURA_STATE_ROOT", str(state))
    monkeypatch.setattr(config.paths, "home_dir_override", None)
    by_name = {condition.name: condition for condition in CONDITIONS}

    async def run() -> tuple[np.ndarray, np.ndarray]:
        runtime = build_runtime(tmp_path / "runtime", seed=11)
        await start_organism(runtime)
        await quiesce_organism(runtime)
        await calibrate_clock(runtime, CONDITIONS, turns=1)
        runtime.freeze_host()
        await runtime.turn_once(by_name["conversation"])
        snapshot = runtime.snapshot()

        async def arm(wait: float) -> np.ndarray:
            runtime.restore(snapshot)
            # Blocking on purpose: the machine is slow, and nothing else runs.
            time.sleep(wait)
            frames = await runtime.turn_once(by_name["memory"])
            return np.vstack([frame.vector() for frame in frames])

        at_once = await arm(0.0)
        # Past the two seconds a phase report stays fresh, with margin.
        after_waiting = await arm(5.0)
        return at_once, after_waiting

    try:
        at_once, after_waiting = asyncio.run(run())
    finally:
        clock = installed_clock()
        if clock is not None:
            clock.uninstall()

    moved = float(np.max(np.abs(after_waiting - at_once)))
    assert moved <= TOLERANCE, f"an arm run after five seconds of waiting moved {moved:.3g}"
