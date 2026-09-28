"""A restore that writes nothing must not report that it restored something.

`_state_is_its_dict` decides whether an object can be put back by writing its
fields, and its docstring names the failure it exists to prevent: a class whose
numbers live in a C base has a `__dict__` too, and restoring it field by field
keeps whatever the last arm wrote.

The heap-type flag used to carry that and no longer does. CPython converted the
standard library's C types to heap types, so `random.Random` passed: its
`__dict__` holds `gauss_next`, its Mersenne state lives in `_random.Random`,
field-by-field wrote `gauss_next`, the restore returned True, and the generator
went on from where the previous arm left it. The workspace draws its somatic
noise from one of these, so it was a floor under every paired measurement —
`workspace.last_winner` and `_somatic_noise` were two of the eleven organs that
did not come back after a restore (docs/THE_SHAM_FLOOR_IS_A_LEAKY_FORK.md).
"""

from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass

import numpy as np
import pytest

from core.subject.copies import _state_is_its_dict, identical
from core.subject.snapshot import _organ_state, _restore_organ


@dataclass
class _Bid:
    content: str = "a"
    priority: float = 0.5


class _Injector:
    """Shaped like `SomaticNoiseInjector`: a seeded generator and a counter."""

    def __init__(self) -> None:
        self.rng = random.Random(7)
        self.count = 0


class _Slotted:
    __slots__ = ("x",)

    def __init__(self) -> None:
        self.x = 1


class _Plain:
    def __init__(self) -> None:
        self.x = 1


def test_a_generator_is_not_restorable_field_by_field() -> None:
    """Its numbers are in the C base, so writing its fields writes one attribute."""
    assert _state_is_its_dict(random.Random()) is False
    assert vars(random.Random()), "if its dict were empty the old check would have held"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(_Bid(), True), (_Plain(), True), (_Slotted(), False), (random.Random(), False)],
)
def test_what_can_be_put_back_by_writing_its_fields(value: object, expected: bool) -> None:
    assert _state_is_its_dict(value) is expected


def test_an_object_holding_a_generator_comes_back() -> None:
    injector = _Injector()

    class _Organ:
        def __init__(self) -> None:
            self.noise = injector

    organ = _Organ()
    saved = _organ_state(organ)
    before = injector.rng.getstate()
    injector.rng.random()
    injector.count = 3
    _restore_organ(organ, saved)
    assert organ.noise.count == 0
    assert organ.noise.rng.getstate() == before, "the generator went on from the last arm"


def test_two_arms_from_one_snapshot_draw_the_same_numbers() -> None:
    """Which is the whole point of a fork."""
    injector = _Injector()

    class _Organ:
        def __init__(self) -> None:
            self.noise = injector

    organ = _Organ()
    saved = _organ_state(organ)
    first = [organ.noise.rng.random() for _ in range(5)]
    _restore_organ(organ, saved)
    second = [organ.noise.rng.random() for _ in range(5)]
    assert first == second


def test_the_shapes_a_fork_carries_all_come_back() -> None:
    class _Organ:
        def __init__(self) -> None:
            self.last_winner = _Bid()
            self.noise = _Injector()
            self.beliefs = {"x": _Bid()}
            self.snapshots = deque([_Bid()], maxlen=8)
            self.learned = np.zeros(4)
            self.lock = threading.Lock()

    organ = _Organ()
    saved = _organ_state(organ)
    organ.last_winner = _Bid("b", 0.9)
    organ.noise.rng.random()
    organ.noise.count = 3
    organ.beliefs["new"] = _Bid()
    organ.snapshots.append(_Bid("d", 0.2))
    organ.learned[:] = 1.0
    _restore_organ(organ, saved)
    fresh = _organ_state(organ)
    missed = [name for name in saved if not identical(saved[name], fresh.get(name))]
    assert missed == [], missed


def test_an_object_keeps_its_identity_through_a_restore() -> None:
    """Something else holds it, so it must be rewound rather than replaced."""
    injector = _Injector()

    class _Organ:
        def __init__(self) -> None:
            self.noise = injector

    organ = _Organ()
    saved = _organ_state(organ)
    organ.noise.count = 3
    _restore_organ(organ, saved)
    assert organ.noise is injector
