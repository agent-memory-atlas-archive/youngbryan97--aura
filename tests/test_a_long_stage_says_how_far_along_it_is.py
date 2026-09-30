"""The grain says how far along it is, with the time left at its own measured rate.

`signature_matrix` is anchors x conditions x actions rollouts and said nothing
between its one opening line and its end. On 29 September a grain probe on three
domains spent 113 minutes there and hit its alarm, and the log could not say
whether it was halfway or stuck. On ten domains the stage is 11,264 rollouts for
the training matrix and 15,360 for the attack — six times the whole 511-cut
sweep — which makes it the longest unwatched stretch of a decisive run.

The estimate is elapsed time over anchors done, a measurement of the run in hand
rather than a figure carried from another one.
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.unit


def _reporter(monkeypatch, said: list[str]):
    import tools.run_subject_core_v25 as runner

    monkeypatch.setattr(runner, "_log", said.append)
    return runner._says_how_far_along("grain, training")


def test_it_reports_about_eleven_times_over_any_length(monkeypatch):
    said: list[str] = []
    far_along = _reporter(monkeypatch, said)
    for done in range(1, 129):
        far_along(done, 128)
    assert 10 <= len(said) <= 12, said


def test_it_says_where_it_is_and_how_long_is_left(monkeypatch):
    said: list[str] = []
    far_along = _reporter(monkeypatch, said)
    far_along(13, 128)
    assert "13/128 anchors" in said[0]
    assert "min spent" in said[0]
    assert "min left at this rate" in said[0]


def test_the_last_anchor_claims_no_time_left(monkeypatch):
    said: list[str] = []
    far_along = _reporter(monkeypatch, said)
    far_along(128, 128)
    assert "128/128 anchors" in said[-1]
    assert "left at this rate" not in said[-1]


def test_a_short_stage_still_reports(monkeypatch):
    said: list[str] = []
    far_along = _reporter(monkeypatch, said)
    far_along(1, 1)
    assert said and "1/1 anchors" in said[0]


def test_the_matrix_takes_the_reporter_and_calls_it_per_anchor():
    from core.subject.v25_grain import signature_matrix

    signature = inspect.signature(signature_matrix)
    assert "progress" in signature.parameters
    assert signature.parameters["progress"].default is None
    body = inspect.getsource(signature_matrix)
    assert "progress(len(rows), len(anchors))" in body


def test_both_grain_stages_are_watched():
    import tools.run_subject_core_v25 as runner

    body = inspect.getsource(runner._learn_grain)
    assert body.count("_says_how_far_along(") == 2
    assert "grain, training" in body and "grain, the attack" in body


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
