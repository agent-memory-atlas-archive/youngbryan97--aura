"""Exact weighted binary fitting with bounded feature materialization."""

import hashlib
import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import numpy as np

BOUNDED_BINARY_FIT_CONTRACT = {
    "schema": "aura.semantic_binary_fit_execution.v1",
    "solver": "blocked_lbfgs",
    "batch_rows": 256,
    "objective": "balanced_weighted_logistic_C10_L2_including_bias",
    "all_rows_used_each_evaluation": True,
    "feature_approximation": False,
    "convergence_required": True,
}

_CHECKPOINT_SCOPE = ContextVar("semantic_binary_fit_checkpoint_scope", default=None)


@contextmanager
def binary_fit_checkpoint_scope(directory: Any, context_identity: Any) -> Iterator[dict[str, Any]]:
    """Bind reusable iterates to one caller's unchanged source/partition custody."""
    if (not isinstance(context_identity, str) or len(context_identity) != 64
            or any(char not in "0123456789abcdef" for char in context_identity)):
        raise ValueError("binary checkpoint scope requires an immutable context identity")
    state = {"directory": Path(directory).resolve(), "context_identity": context_identity,
             "records": [], "paths": {}}
    token = _CHECKPOINT_SCOPE.set(state)
    try:
        yield state
    finally:
        _CHECKPOINT_SCOPE.reset(token)


def binary_fit_checkpoint_receipt() -> Any:
    from core.learning.semantic_fit_checkpoint import fit_identity

    state = _CHECKPOINT_SCOPE.get()
    if state is None:
        return None
    body = {"schema": "aura.semantic_binary_fit_checkpoints.v1",
            "context_identity": state["context_identity"],
            "records": [dict(record) for record in state["records"]],
            "final_archives": {identity: hashlib.sha256(path.read_bytes()).hexdigest()
                               for identity, path in sorted(state["paths"].items())},
            "optimizer_resume": "accepted_iterate_with_fresh_lbfgs_history",
            "cached_convergence_is_authority": False, "serving_authority": False}
    return {**body, "receipt_sha256": fit_identity(body)}


def _objective_checkpoint(features: Any, labels: Any, sample_weight: Any, tolerance: float) -> Any:
    from core.learning.semantic_fit_checkpoint import ObjectiveFitCheckpoint, fit_identity

    state = _CHECKPOINT_SCOPE.get()
    if state is None:
        return None
    import scipy
    import sklearn

    data_hash = hashlib.sha256()
    batch_rows = BOUNDED_BINARY_FIT_CONTRACT["batch_rows"]
    for start in range(0, features.shape[0], batch_rows):
        stop = min(start + batch_rows, features.shape[0])
        rows = np.asarray(features[start:stop], dtype="<f8")
        if rows.shape != (stop - start, features.shape[1]) or not np.all(np.isfinite(rows)):
            raise ValueError("binary checkpoint feature batch differs")
        data_hash.update(rows.tobytes(order="C"))
    identity = fit_identity({"contract": BOUNDED_BINARY_FIT_CONTRACT,
        "context": state["context_identity"], "shape": features.shape,
        "features_sha256": data_hash.hexdigest(), "labels": np.asarray(labels, dtype=np.int64),
        "sample_weight": np.ones(len(labels), dtype=np.float64) if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float64),
        "gtol": tolerance, "ftol": 1e-12,
        "implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_implementation": hashlib.sha256(Path(sys.modules[
            ObjectiveFitCheckpoint.__module__].__file__).read_bytes()).hexdigest(),
        "python": sys.version, "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__})
    path = state["directory"] / (identity + ".npz")
    state["paths"][identity] = path
    return ObjectiveFitCheckpoint(path, identity, np.zeros(features.shape[1] + 1, dtype=np.float64))


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
    checkpoint = _objective_checkpoint(features, labels, sample_weight, tolerance)
    start = np.zeros(features.shape[1] + 1, dtype=np.float64)
    saved = None if checkpoint is None else checkpoint.load()
    resumed_iterations = 0
    if saved is not None:
        start, resumed_iterations = saved["weight"], saved["iterations"]
    iterations = 0
    if progress is not None:
        progress({"stage": "binary_fit_start", "rows": features.shape[0], "width": features.shape[1],
                  "solver": "blocked_lbfgs", "max_iter": max_iter, "tolerance": tolerance,
                  "resumed_iterations": resumed_iterations,
                  "checkpoint_identity": None if checkpoint is None else checkpoint.identity})
    def advanced(parameters: Any) -> None:
        nonlocal iterations
        iterations += 1
        if checkpoint is not None and iterations % 25 == 0:
            checkpoint.save(parameters, resumed_iterations + iterations)
        if progress is not None and iterations % 25 == 0:
            progress({"stage": "binary_fit_iteration", "iteration": iterations,
                      "resumed_iterations": resumed_iterations, "solver": "blocked_lbfgs"})
    result = minimize(objective, start,
                      method="L-BFGS-B", jac=True, callback=advanced,
                      options={"maxiter": max_iter, "gtol": tolerance, "ftol": 1e-12})
    finite = (np.asarray(result.x).shape == start.shape and np.all(np.isfinite(result.x))
              and math.isfinite(result.fun))
    if checkpoint is not None and finite:
        checkpoint.save(result.x, resumed_iterations + iterations, converged=bool(result.success))
    if not result.success or not finite:
        raise ValueError(f"bounded binary fit did not converge: {result.message}")
    if checkpoint is not None:
        from core.learning.semantic_fit_checkpoint import fit_identity
        _CHECKPOINT_SCOPE.get()["records"].append({
            "objective_identity": checkpoint.identity, "resumed_iterations": resumed_iterations,
            "iterations": int(result.nit), "max_iter": max_iter, "tolerance": tolerance,
            "initial_parameters_sha256": fit_identity(start),
            "final_parameters_sha256": fit_identity(np.asarray(result.x, dtype=np.float64)),
            "objective": float(result.fun), "converged": True,
            "convergence_message": str(result.message)})
    if progress is not None:
        progress({"stage": "binary_fit_complete", "iterations": result.nit,
                  "objective": float(result.fun), "solver": "blocked_lbfgs"})
    return np.asarray(result.x[:-1], dtype=np.float32), float(result.x[-1])
