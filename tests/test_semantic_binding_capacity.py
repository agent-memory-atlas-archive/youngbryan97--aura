"""Exact surrogate math, measured invariance, and source-only capacity choice."""

import numpy as np
import pytest

from core.learning.semantic_binding_capacity import reduced_rank_binding_fit
from core.learning.semantic_binding_invariance import fit_binding_nuisance_projection
from tools.semantic_binding_capacity_plan import source_capacity_selection


def test_anisotropic_reduced_rank_fit_reaches_its_exact_observed_error_bound():
    random = np.random.default_rng(19)
    x = random.normal(size=(30, 5)) * np.array([.01, 1., 100., .1, 10.])
    y = random.normal(size=(30, 4))
    coefficient, receipt = reduced_rank_binding_fit(x, y, rank=2)
    assert np.linalg.matrix_rank(coefficient) == 2
    assert receipt["actual_squared_error"] == pytest.approx(receipt["minimum_squared_error"], abs=1e-8)
    assert not receipt["native_decoder_ceiling"]
    _full, full = reduced_rank_binding_fit(x, y, rank=8)
    assert full["rank_tail_error"] == 0. and full["minimum_squared_error"] <= receipt["minimum_squared_error"]


def test_deleted_linear_information_remains_inaccessible_not_a_native_ceiling():
    coefficient, receipt = reduced_rank_binding_fit(np.zeros((4, 3)), np.ones((4, 2)), rank=2)
    assert np.array_equal(coefficient, np.zeros((3, 2)))
    assert receipt["inaccessible_error"] == 8.


def test_projection_estimates_only_changed_source_directions_and_roundtrips():
    from core.learning.semantic_binding_invariance import BindingNuisanceProjection

    left = np.array([[1., 2., 3.], [2., 5., 4.]])
    right = left + np.array([0., 7., 0.])
    projection = fit_binding_nuisance_projection({"a": (left, right)}, training_ids=("a",), excluded_ids=("held",), rank=1)
    assert np.allclose(projection(left), projection(right))
    assert np.allclose(projection(left)[:, [0, 2]], left[:, [0, 2]])
    assert BindingNuisanceProjection.from_dict(projection.to_dict()).to_dict() == projection.to_dict()
    with pytest.raises(ValueError, match="disjoint"):
        fit_binding_nuisance_projection({"held": (left, right)}, training_ids=("held",), excluded_ids=("held",))


def test_capacity_choice_requires_measured_retention_not_just_lower_loss():
    valid = {"correct_source_ids": ["a", "b"], "measured_source_ids": ["a", "b"],
             "selected_step": 4, "checkpoint_receipt_sha256": "a" * 64,
             "trainable_parameters": 100, "source_calibration_loss": .3, "selection_uses_held_targets": False}
    result = source_capacity_selection([valid, {**valid, "correct_source_ids": ["b"], "source_calibration_loss": .1},
                                        {**valid, "selected_step": None}],
                                      baseline_correct_ids=("a",), source_calibration_ids=("a", "b"))
    assert result["selected_index"] == 0 and not result["promotion_authority"]
    assert result["decisions"][1]["reasons"] == ["baseline_source_regression"]
