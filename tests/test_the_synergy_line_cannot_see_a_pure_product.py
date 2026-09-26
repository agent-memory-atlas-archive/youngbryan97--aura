"""The synergy line cannot see a pure product; a Kraskov estimate can, and rejects a sum.

docs/SYNERGY_KNOWN_ANSWERS.md. A zero-mean product of two independent sources
has no rank correlation with either of them, and the v3 line reads synergy
through a Gaussian copula, which keeps only rank correlations. A Kraskov
estimator does see the product, scores a threshold of sums as synergy, which it
is, and scores a separable sum below zero. These pin all of that on one seed of
the study's toy, so a change to the line that lets it see a product flips a test
instead of passing unnoticed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from core.subject.synergy import _components, _copula_normal, kraskov_synergy_value, synergy

pytestmark = pytest.mark.unit

_TOOL = Path(__file__).resolve().parents[1] / "tools" / "synergy_known_answers.py"
_SPEC = importlib.util.spec_from_file_location("synergy_known_answers", _TOOL)
study = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(study)

SEED = 3


def _ksg_synergy(kind: str) -> float:
    recording = study.build(kind, SEED)
    following = recording.domain("D")
    a, b, y = (
        _copula_normal(_components(block))
        for block in (recording.domain("W")[:-1], recording.domain("A")[:-1], following[1:] - following[:-1])
    )
    return kraskov_synergy_value(a, b, y, np.random.default_rng(SEED))[0]


def test_the_v3_line_does_not_register_a_pure_product():
    report = synergy(study.build("product", SEED), "W", "A", "D", seed=SEED, of="change")
    assert not report.passes_v3
    assert report.synergy <= report.raw_null_q99 + 1e-3


def test_a_kraskov_estimate_sees_the_product_and_nothing_where_there_is_nothing():
    assert _ksg_synergy("product") > 0.1
    assert _ksg_synergy("none") < 0.0


def test_a_threshold_of_sums_is_synergy_and_a_separable_sum_is_not():
    """Which of four sums is largest depends on W and A together; a sum does not."""
    assert _ksg_synergy("additive") > 0.03
    assert _ksg_synergy("separable") < 0.0


def test_the_qualifying_rule_is_the_designs():
    assert study.qualifies({"product": 4, "mixed": 4, "separable": 1, "none": 1}, 5)
    assert not study.qualifies({"product": 5, "mixed": 5, "separable": 5, "none": 1}, 5)
    assert not study.qualifies({"product": 0, "mixed": 2, "separable": 0, "none": 0}, 5)


def test_the_kraskov_line_passes_a_product_and_not_a_separable_sum():
    """The line the study qualified, as the battery reports it, on one seed with few draws."""
    from core.subject.synergy import kraskov_synergy

    product = kraskov_synergy(study.build("product", SEED), "W", "A", "D", seed=SEED, draws=20, clocks_out=False)
    separable = kraskov_synergy(study.build("separable", SEED), "W", "A", "D", seed=SEED, draws=20, clocks_out=False)
    assert product.passes, product
    assert product.synergy > product.additive_bar
    assert not separable.passes, separable
    assert product.as_dict()["estimator"] == "kraskov_k3_max_norm"


def test_the_additive_account_holds_a_sum_and_not_a_product():
    from core.subject.synergy import additive_account

    rng = np.random.default_rng(SEED)
    a, b = rng.normal(size=(2400, 2)), rng.normal(size=(2400, 2))
    total = (a[:, :1] + b[:, :1] ** 2) + 0.1 * rng.normal(size=(2400, 1))
    product = a[:, :1] * b[:, :1] + 0.1 * rng.normal(size=(2400, 1))
    explained = 1.0 - np.var(total - additive_account(a, b, total)) / np.var(total)
    missed = 1.0 - np.var(product - additive_account(a, b, product)) / np.var(product)
    assert explained > 0.9
    assert missed < 0.05
