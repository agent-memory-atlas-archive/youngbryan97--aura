"""Exact weighted binary fitting with bounded feature materialization."""

import math

import numpy as np
from typing import Any


BOUNDED_BINARY_FIT_CONTRACT = {
    "schema": "aura.semantic_binary_fit_execution.v1",
    "solver": "blocked_lbfgs",
    "batch_rows": 256,
    "objective": "balanced_weighted_logistic_C10_L2_including_bias",
    "all_rows_used_each_evaluation": True,
    "feature_approximation": False,
    "convergence_required": True,
}


class BinaryFeatureRows:
    """Keep ordered row references until the solver requests a bounded batch."""

    def __init__(self, rows: Any) -> None:
        self._rows = tuple(rows)
        if not self._rows or any(row.ndim != 1 or row.shape != self._rows[0].shape for row in self._rows):
            raise ValueError("binary feature row geometry differs")
        self.shape = (len(self._rows), self._rows[0].size)
        self.ndim = 2

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, (int, np.integer)):
            return self._rows[index]
        return np.stack(self._rows[index])


def binary_objective(features: Any, labels: Any, sample_weight: Any, *, batch_rows: int=256) -> Any:
    """Return the liblinear objective, including its regularized unit bias."""
    from scipy.special import expit
    from sklearn.utils.class_weight import compute_class_weight

    labels = np.asarray(labels)
    if (getattr(features, "ndim", None) != 2 or len(features.shape) != 2
            or features.shape[0] < 2 or features.shape[1] < 1
            or labels.shape != (features.shape[0],) or set(np.unique(labels)) != {0, 1}
            or type(batch_rows) is not int or not 1 <= batch_rows <= 4096):
        raise ValueError("bounded binary fit needs finite two-class row geometry")
    weights = (np.ones(len(labels), dtype=np.float64) if sample_weight is None
               else np.asarray(sample_weight, dtype=np.float64))
    if (weights.shape != labels.shape or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("bounded binary fit sample weights differ")
    class_weights = compute_class_weight("balanced", classes=np.asarray([0, 1]),
                                          y=labels, sample_weight=weights)
    if not np.all(np.isfinite(class_weights)):
        raise ValueError("bounded binary fit class weights are nonfinite")
    weights = weights * np.where(labels == 1, class_weights[1], class_weights[0])

    def objective(parameters: Any) -> tuple[Any, Any]:
        parameters = np.asarray(parameters, dtype=np.float64)
        if parameters.shape != (features.shape[1] + 1,) or not np.all(np.isfinite(parameters)):
            raise ValueError("bounded binary fit parameters differ")
        value = .5 * float(parameters @ parameters)
        gradient = parameters.copy()
        for start in range(0, len(labels), batch_rows):
            stop = min(start + batch_rows, len(labels))
            rows = np.asarray(features[start:stop], dtype=np.float64)
            if rows.shape != (stop - start, features.shape[1]) or not np.all(np.isfinite(rows)):
                raise ValueError("bounded binary fit feature batch differs")
            scores = rows @ parameters[:-1] + parameters[-1]
            selected_weights = weights[start:stop]
            value += 10 * float(selected_weights @ np.logaddexp(
                0, np.where(labels[start:stop] == 1, -scores, scores)))
            residual = 10 * selected_weights * (expit(scores) - labels[start:stop])
            gradient[:-1] += rows.T @ residual
            gradient[-1] += float(np.sum(residual))
        if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise ValueError("bounded binary fit objective is nonfinite")
        return value, gradient

    return objective


def fit_bounded_binary_head(
    features: Any,
    labels: Any,
    *,
    sample_weight: Any=None,
    max_iter: int=1000,
    tolerance: float=0.0001,
    progress: Any=None,
) -> tuple[Any, float]:
    from scipy.optimize import minimize

    if type(max_iter) is not int or max_iter < 1 or not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("bounded binary fit requires a finite convergence contract")
    objective = binary_objective(features, labels, sample_weight,
                                 batch_rows=BOUNDED_BINARY_FIT_CONTRACT["batch_rows"])
    iterations = 0
    if progress is not None:
        progress({"stage": "binary_fit_start", "rows": features.shape[0], "width": features.shape[1],
                  "solver": "blocked_lbfgs"})
    def advanced(_parameters: Any) -> None:
        nonlocal iterations
        iterations += 1
        if progress is not None and iterations % 25 == 0:
            progress({"stage": "binary_fit_iteration", "iteration": iterations, "solver": "blocked_lbfgs"})
    result = minimize(objective, np.zeros(features.shape[1] + 1, dtype=np.float64),
                      method="L-BFGS-B", jac=True, callback=advanced,
                      options={"maxiter": max_iter, "gtol": tolerance, "ftol": 1e-12})
    if not result.success or not np.all(np.isfinite(result.x)) or not math.isfinite(result.fun):
        raise ValueError(f"bounded binary fit did not converge: {result.message}")
    if progress is not None:
        progress({"stage": "binary_fit_complete", "iterations": result.nit,
                  "objective": float(result.fun), "solver": "blocked_lbfgs"})
    return np.asarray(result.x[:-1], dtype=np.float32), float(result.x[-1])
