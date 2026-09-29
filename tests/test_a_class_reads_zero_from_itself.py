"""A percept class presented twice from the same anchors reads zero from itself.

The internal geometry compares classes forked from the same anchors, so row r
of every class belongs to one (anchor, condition). Cross-fitted without that,
a test row's twin from the other class sat in the training fold with the other
label, and a class read about 0.05 from itself on seed 7: the estimator's floor
on two identical samples, not her noise. Paired, classes are compared on the
presentations both have and folds keep an anchor's rows together.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.subject.content_runtime import ClassSamples, internal_geometry

pytestmark = pytest.mark.unit


def _class(name: str, futures: np.ndarray, contexts: np.ndarray, keys: list[tuple[int, int]]) -> ClassSamples:
    rows = ClassSamples(name, [f for f in futures], [frozenset() for _ in futures], [c for c in contexts])
    rows.futures_b = [f.copy() for f in futures]
    rows.retrieved_b = [frozenset() for _ in futures]
    rows.keys = list(keys)
    return rows


def _bank(anchors: int = 12, conditions: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    contexts = rng.normal(size=(anchors, 8))
    keys = [(a, c) for a in range(anchors) for c in range(conditions)]
    ctx = np.asarray([contexts[a] for a, _c in keys])
    future = rng.normal(size=(len(keys), 6))
    return keys, ctx, future


def test_two_identical_classes_and_each_class_against_itself_read_zero() -> None:
    keys, ctx, future = _bank()
    samples = {"a": _class("a", future, ctx, keys), "b": _class("b", future.copy(), ctx, keys)}
    classes = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    distances, floor = internal_geometry(samples, classes, paired=True)
    assert distances[(0, 1)] == 0.0
    assert floor[(0, 0)] == 0.0 and floor[(1, 1)] == 0.0


def test_a_class_that_missed_a_presentation_is_compared_on_the_ones_both_have() -> None:
    keys, ctx, future = _bank()
    keep = [i for i in range(len(keys)) if i != 5]
    samples = {
        "a": _class("a", future, ctx, keys),
        "b": _class("b", future[keep], ctx[keep], [keys[i] for i in keep]),
    }
    classes = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    distances, _floor = internal_geometry(samples, classes, paired=True)
    assert distances[(0, 1)] == 0.0


def test_two_classes_that_differ_are_apart() -> None:
    keys, ctx, future = _bank()
    moved = future.copy()
    moved[:, :2] += 1.5
    samples = {"a": _class("a", future, ctx, keys), "b": _class("b", moved, ctx, keys)}
    classes = [SimpleNamespace(name="a"), SimpleNamespace(name="b")]
    distances, _floor = internal_geometry(samples, classes, paired=True)
    assert distances[(0, 1)] > 0.1
