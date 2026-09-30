"""Bounded batches preserve the complete weighted binary objective."""

import numpy as np
import pytest

from core.learning.semantic_bounded_binary_fit import BinaryFeatureRows, binary_objective
from core.learning.semantic_program_transducer import _fit_binary_head
from core.learning.semantic_relation_tissue import DirectionalFeatureRows


@pytest.mark.parametrize("weighted", [False, True])
def test_bounded_fit_matches_existing_coefficients_with_regularized_bias(weighted):
    rng = np.random.default_rng(271828)
    x = rng.normal(size=(160, 12)).astype(np.float32)
    y = (x[:, 0] + .2 * x[:, 1] + rng.normal(size=160) > .8).astype(np.int8)
    weights = rng.uniform(.2, 2., len(y)) if weighted else None
    kwargs = {"sample_weight": weights, "tolerance": 1e-9}
    expected, expected_bias = _fit_binary_head(x, y, **kwargs)
    actual, actual_bias = _fit_binary_head(BinaryFeatureRows(x), y, solver="blocked_lbfgs", **kwargs)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    assert actual_bias == pytest.approx(expected_bias, abs=2e-5)


def test_bounded_objective_and_gradient_equal_full_matrix_and_finite_differences():
    from scipy.special import expit
    rng = np.random.default_rng(1729)
    x = rng.normal(size=(37, 7)).astype(np.float32)
    y = np.asarray([0] * 25 + [1] * 12)
    sw = rng.uniform(.2, 2., len(y))
    parameters = rng.normal(size=8)
    objective = binary_objective(BinaryFeatureRows(x), y, sw, batch_rows=5)
    value, gradient = objective(parameters)
    effective = sw * np.where(y == 1, np.sum(sw) / (2 * np.sum(sw[y == 1])),
                              np.sum(sw) / (2 * np.sum(sw[y == 0])))
    augmented = np.column_stack((x.astype(np.float64), np.ones(len(x))))
    scores = augmented @ parameters
    expected_value = .5 * (parameters @ parameters) + 10 * effective @ (np.logaddexp(0, scores) - y * scores)
    expected_gradient = parameters + 10 * augmented.T @ (effective * (expit(scores) - y))
    assert value == pytest.approx(expected_value, abs=1e-10)
    np.testing.assert_allclose(gradient, expected_gradient, atol=1e-10)
    step = 1e-5
    numerical = []
    for index in range(len(parameters)):
        delta = np.zeros_like(parameters)
        delta[index] = step
        numerical.append((objective(parameters + delta)[0] - objective(parameters - delta)[0]) / (2 * step))
    np.testing.assert_allclose(gradient, numerical, atol=1e-7, rtol=1e-7)


def test_relation_factorization_is_exact_and_solver_never_requests_a_full_matrix():
    rng = np.random.default_rng(42)
    rows = DirectionalFeatureRows()
    for _ in range(600):
        rows.append(rng.normal(size=8).astype(np.float32), rng.normal(size=8).astype(np.float32))
    dense = rows[:]
    y = (dense[:, 0] > .5).astype(np.int8)
    requested = []
    class BoundedView:
        shape, ndim = rows.shape, rows.ndim
        def __getitem__(self, indices):
            assert isinstance(indices, slice)
            count = indices.stop - indices.start
            assert 0 < count <= 256
            requested.append(count)
            return rows[indices]
    expected, old_bias = _fit_binary_head(dense, y, tolerance=1e-8)
    actual, bias = _fit_binary_head(BoundedView(), y, solver="blocked_lbfgs", tolerance=1e-8)
    assert max(requested) == 256
    np.testing.assert_allclose(actual, expected, atol=3e-5, rtol=3e-5)
    assert bias == pytest.approx(old_bias, abs=3e-5)


def test_unfinished_bounded_fit_is_rejected_and_success_emits_actual_progress():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(160, 12)).astype(np.float32)
    y = (x[:, 0] > 0).astype(np.int8)
    with pytest.raises(ValueError, match="did not converge"):
        _fit_binary_head(x, y, solver="blocked_lbfgs", max_iter=1)
    progress = []
    _fit_binary_head(x, y, solver="blocked_lbfgs", progress=progress.append)
    assert progress[0]["stage"] == "binary_fit_start"
    assert progress[-1]["stage"] == "binary_fit_complete"
    assert progress[-1]["iterations"] > 0 and np.isfinite(progress[-1]["objective"])


@pytest.mark.parametrize("fault", ["single_class", "wrong_labels", "negative_weights", "nan_weights", "nan_features"])
def test_invalid_binary_inputs_cannot_produce_a_fitted_head(fault):
    x = np.ones((4, 2), dtype=np.float32)
    y = np.asarray([0, 0, 1, 1])
    weights = np.ones(4)
    if fault == "single_class":
        y[:] = 0
    elif fault == "wrong_labels":
        y = y[:3]
    elif fault == "negative_weights":
        weights[0] = -1
    elif fault == "nan_weights":
        weights[0] = np.nan
    elif fault == "nan_features":
        x[0, 0] = np.nan
    with pytest.raises(ValueError):
        _fit_binary_head(x, y, sample_weight=weights, solver="blocked_lbfgs")
