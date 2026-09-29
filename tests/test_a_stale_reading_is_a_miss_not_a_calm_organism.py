"""A substrate reading older than its own freshness bound is a miss.

Phase 30 of the ISC completion list is about not turning measurement failures
into plausible zeros. Genuine zero, absent organ and failed reader are three
states the recording already tells apart (P30.2 to P30.4). A stale reading is
the fourth and the quietest: the substrate publishes how old its snapshot is and
returns its safe defaults when it cannot answer now, and those defaults are
plausible numbers indistinguishable from a settled state. Nine of recurrent
cognition's columns would read as a calm organism exactly when its dynamics had
stopped.

Recorded as a miss, which is what the battery invalidates a criterion on.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.subject.state import recording_misses

pytestmark = pytest.mark.unit

SOURCE = "organ:substrate.get_substrate_affect"


class _Substrate:
    """Her substrate, answering with a snapshot of a given age."""

    def __init__(self, *, stale: float, age: float) -> None:
        self._stale, self._age = stale, age

    def get_substrate_affect(self):
        return {
            "valence": 0.0,
            "arousal": 0.5,
            "dominance": 0.5,
            "energy": 0.7,
            "volatility": 0.0,
            "snapshot_stale": self._stale,
            "snapshot_age_s": self._age,
        }

    def get_status(self):
        return {"focus": 1.0, "curiosity": 1.0, "frustration": 0.0}


def _read(stale: float, age: float) -> dict[str, str]:
    from core.subject.state_readers import _read_C

    organs = SimpleNamespace(
        substrate=_Substrate(stale=stale, age=age),
        mesh=None, field=None, phi_core=None, worth=None,
    )
    state = SimpleNamespace()
    with recording_misses() as misses:
        try:
            _read_C(state, organs)
        # A reader that cannot walk this stub state still records its misses;
        # what is under test is whether staleness is one of them.
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return misses


def test_a_stale_snapshot_is_recorded_as_a_miss():
    misses = _read(stale=1.0, age=4.25)
    assert SOURCE in misses, misses
    assert "4.25s old" in misses[SOURCE]


def test_a_fresh_snapshot_is_not():
    assert SOURCE not in _read(stale=0.0, age=0.01)


def test_the_bound_is_the_substrate_s_own_and_not_a_number_chosen_here():
    # It reports staleness as a ratio against its own freshness bound, so the
    # reader compares with one rather than with a duration of its choosing.
    assert SOURCE not in _read(stale=0.99, age=99.0)
    assert SOURCE in _read(stale=1.0, age=0.001)


def test_the_reason_names_how_old_the_reading_was():
    misses = _read(stale=3.0, age=12.5)
    assert "12.5s old" in misses[SOURCE]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
