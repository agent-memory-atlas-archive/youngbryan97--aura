"""A worker's launch challenge is stamped and read on the machine's clock.

The parent signs a challenge for each worker it spawns, valid for five
minutes, and the worker refuses to start if the challenge is not current. The
parent stamped it with `time.time()`. In a measurement run that is the run's
own clock (core/subject/clock.py), which advances a second a turn and falls
behind the machine's; the worker is another process and reads the machine's.
On the reports-ground run of 26-27 September her cortex died during the
baseline, and every respawn after it died at start with
"worker_capture_launch_challenge_not_current", until the run hit its eight-hour
bound with no answer from her.
"""

from __future__ import annotations

import time

from core.brain.llm.latent_cortex.worker_capture_identity import (
    build_worker_capture_identity,
    build_worker_capture_launch_authority,
)


def test_a_challenge_issued_under_a_run_clock_is_current_in_the_worker(monkeypatch):
    behind = time.time() - 3 * 3600
    with monkeypatch.context() as run:
        run.setattr(time, "time", lambda: behind)
        authority = build_worker_capture_launch_authority()
    identity = build_worker_capture_identity(
        worker_boot_id="a" * 32,
        launch_challenge=authority.challenge,
    )
    assert identity is not None


def test_the_parent_reads_it_on_the_same_clock(monkeypatch):
    authority = build_worker_capture_launch_authority()
    behind = time.time() - 3 * 3600
    monkeypatch.setattr(time, "time", lambda: behind)
    identity = build_worker_capture_identity(
        worker_boot_id="b" * 32,
        launch_challenge=authority.challenge,
    )
    assert identity is not None
