"""The synergy line cannot see a pure product, and what can see it cannot tell it from a sum.

docs/SYNERGY_KNOWN_ANSWERS.md. A zero-mean product of two independent sources
has no rank correlation with either of them, and the v3 line reads synergy
through a Gaussian copula, which keeps only rank correlations. A Kraskov
estimator does see the product, and scores a plain sum as synergy too, which is
what MMI synergy does with a sum. These pin both on one seed of the study's toy,
so a change to the line that lets it see a product flips a test instead of
passing unnoticed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from core.subject.synergy import _components, _copula_normal, synergy

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
    return study._ksg_synergy(a, b, y, np.random.default_rng(SEED))[0]


def test_the_v3_line_does_not_register_a_pure_product():
    report = synergy(study.build("product", SEED), "W", "A", "D", seed=SEED, of="change")
    assert not report.passes_v3
    assert report.synergy <= report.raw_null_q99 + 1e-3


def test_a_kraskov_estimate_sees_the_product_and_nothing_where_there_is_nothing():
    assert _ksg_synergy("product") > 0.1
    assert _ksg_synergy("none") < 0.0


def test_a_kraskov_estimate_also_scores_a_sum_as_synergy():
    assert _ksg_synergy("additive") > 0.03


def test_the_qualifying_rule_is_the_designs():
    assert study.qualifies({"product": 4, "mixed": 4, "additive": 1, "none": 1}, 5)
    assert not study.qualifies({"product": 5, "mixed": 5, "additive": 5, "none": 1}, 5)
    assert not study.qualifies({"product": 0, "mixed": 2, "additive": 0, "none": 0}, 5)
