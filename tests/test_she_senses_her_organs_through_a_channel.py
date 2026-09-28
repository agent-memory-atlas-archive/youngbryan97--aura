"""An organ's own number is not something any part of her can read without a channel.

`core/subject/afferent.py`. A hundred and fifty-seven of the 375 columns of K_t
are read straight off a live organ. Every channel in a body is a low pass
filter, so what arrives centrally is the recent history of a signal rather than
its instantaneous truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.subject.afferent import Afferent, afferent_turns, organ_sourced, sensed


class _HasSurface:
    def __init__(self, surface: object) -> None:
        self.afferent = surface


def test_off_unless_a_run_asks_for_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AURA_AFFERENT_TURNS", raising=False)
    assert afferent_turns() == 0.0
    assert not Afferent(33.0).on
    monkeypatch.setenv("AURA_AFFERENT_TURNS", "1")
    assert Afferent(33.0).on
    monkeypatch.setenv("AURA_AFFERENT_TURNS", "not a number")
    assert afferent_turns() == 0.0


def test_a_surface_that_is_off_returns_the_organ_s_own_array() -> None:
    surface = _surface(turns=0.0)
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(surface.sense("C", values, ("a", "b", "c")), values)


def _surface(turns: float = 1.0) -> Afferent:
    """A surface that owns both toy columns, so a test does not depend on the schema."""
    return Afferent(33.0, turns=turns, mine={"C": frozenset({"a", "b"})})


def test_the_first_reading_is_sensed_as_it_came() -> None:
    surface = _surface()
    values = np.array([1.0, 2.0])
    assert np.allclose(surface.sense("C", values, ("a", "b")), values)


def test_a_column_the_membrane_already_carries_is_not_sensed_as_well() -> None:
    """Two surfaces on one column would give it two time constants.

    `core/runtime/state_membrane.py` carries every column that is a plain number
    at a writable path, and a reader returns its whole domain. So the surface
    touches only what the schema declares as coming from an organ, and the two
    arms of the campaign stay distinguishable.
    """
    surface = Afferent(33.0, turns=1.0, mine={"C": frozenset({"organ_one"})})
    surface.sense("C", np.array([0.0, 0.0]), ("organ_one", "plain_one"))
    out = surface.sense("C", np.array([1.0, 1.0]), ("organ_one", "plain_one"))
    assert out[0] < 1.0, "the organ column was not sensed"
    assert out[1] == pytest.approx(1.0), "a column the membrane carries was sensed too"


def test_every_domain_it_owns_is_one_a_reader_reads_off_an_organ() -> None:
    from core.subject.state import _ORGAN_READERS

    mine = organ_sourced()
    assert set(mine) == set(_ORGAN_READERS)
    assert sum(len(names) for names in mine.values()) > 100


def test_a_step_arrives_over_the_channel_s_own_time() -> None:
    surface = _surface()
    surface.sense("C", np.array([0.0]), ("a",))
    out = surface.sense("C", np.array([1.0]), ("a",))
    assert 0.0 < out[0] < 1.0
    for _ in range(200):
        out = surface.sense("C", np.array([1.0]), ("a",))
    assert out[0] == pytest.approx(1.0, abs=0.01)


def test_each_column_is_its_own_channel() -> None:
    surface = _surface()
    surface.sense("C", np.array([0.0, 10.0]), ("a", "b"))
    out = surface.sense("C", np.array([1.0, 10.0]), ("a", "b"))
    assert out[0] != 0.0
    assert out[1] == pytest.approx(10.0)
    assert surface.of("C.a") == pytest.approx(out[0])
    assert surface.of("C.b") == pytest.approx(10.0)
    assert surface.of("C.never seen") is None


def test_a_reader_and_a_schema_that_disagree_are_left_to_the_width_check() -> None:
    surface = _surface()
    values = np.array([1.0, 2.0, 3.0])
    assert np.allclose(surface.sense("C", values, ("a", "b")), values)


def test_a_state_with_no_surface_records_the_organ() -> None:
    values = np.array([1.0, 2.0])
    assert np.allclose(sensed(object(), "C", values, ("a", "b")), values)
    assert np.allclose(sensed(_HasSurface("not a surface"), "C", values, ("a", "b")), values)


def test_forgetting_leaves_a_fork_without_a_body_it_did_not_live_in() -> None:
    surface = _surface()
    surface.sense("C", np.array([4.0]), ("a",))
    surface.forget()
    assert surface.of("C.a") is None


def test_the_state_reading_takes_the_sensed_array() -> None:
    import inspect

    from core.subject import state as schema

    source = inspect.getsource(schema.read_core_state)
    assert "sensed(state, key, organ_reader(state, kit)" in source


def test_the_driver_attaches_it_before_it_settles() -> None:
    import inspect

    from core.subject.driver import SubjectRuntime

    settle = inspect.getsource(SubjectRuntime._settle_membrane)
    assert "_attach_afferent()" in settle
    assert list(inspect.signature(SubjectRuntime._attach_afferent).parameters) == ["self"]
