"""Controlled nuisance projection; no positional-subspace assumption or oracle."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BindingNuisanceProjection:
    basis: np.ndarray
    training_ids: tuple[str, ...]

    def __post_init__(self):
        basis = np.array(self.basis, dtype=np.float64, copy=True)
        if (basis.ndim != 2 or basis.shape[0] < 1 or basis.shape[1] >= basis.shape[0]
                or not np.isfinite(basis).all()
                or not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-8)
                or not self.training_ids or len(set(self.training_ids)) != len(self.training_ids)):
            raise ValueError("binding nuisance projection needs a measured orthonormal source basis")
        basis.setflags(write=False)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "training_ids", tuple(self.training_ids))

    def __call__(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != self.basis.shape[0] or not np.isfinite(values).all():
            raise ValueError("binding nuisance projection evidence differs")
        return values - (values @ self.basis) @ self.basis.T

    def to_dict(self):
        return {"schema": "aura.binding_nuisance_projection.v1", "basis": self.basis.tolist(),
                "training_ids": list(self.training_ids), "rank": self.basis.shape[1]}

    @classmethod
    def from_dict(cls, value):
        projection = cls(value["basis"], tuple(value["training_ids"]))
        if projection.to_dict() != value:
            raise ValueError("binding nuisance projection contract differs")
        return projection


def fit_binding_nuisance_projection(pairs, *, training_ids, excluded_ids=(), rank=8):
    """Pairs must be source-witnessed meaning-preserving interventions.

    This fits changed directions, not a claim that all such directions are
    nuisance-only. Source calibration and causal controls must establish that
    removing the fitted basis does not destroy the relation signal.
    """
    training_ids, excluded_ids = tuple(training_ids), frozenset(excluded_ids)
    if (not training_ids or len(set(training_ids)) != len(training_ids)
            or set(training_ids) & excluded_ids or set(pairs) != set(training_ids)
            or type(rank) is not int or rank < 1):
        raise ValueError("nuisance projection needs disjoint source-only interventions")
    differences = []
    for identity in training_ids:
        left, right = (np.asarray(value, dtype=np.float64) for value in pairs[identity])
        if (left.shape != right.shape or left.ndim < 1 or min(left.shape) < 1
                or not np.isfinite(left).all() or not np.isfinite(right).all()):
            raise ValueError("nuisance intervention state geometry differs")
        differences.extend((left - right).reshape(-1, left.shape[-1]))
    values = np.stack(differences)
    if rank >= values.shape[1]:
        raise ValueError("nuisance removal must retain at least one representation dimension")
    _, singular, vectors = np.linalg.svd(values, full_matrices=False)
    measured_rank = np.count_nonzero(singular > singular[0] * max(values.shape) * np.finfo(float).eps)
    basis = vectors[:min(rank, measured_rank)].T
    return BindingNuisanceProjection(basis, training_ids)


def reverse_nuisance_gradient(value, strength):
    """Forward identity, reversed feature gradient; classifier learns normally."""
    import math

    import mlx.core as mx

    if not math.isfinite(strength) or strength < 0:
        raise ValueError("nuisance reversal needs nonnegative finite strength")
    return (1. + strength) * mx.stop_gradient(value) - strength * value


def binding_choice_scale_penalty(scores, positives):
    """Squared derivative of source choice risk w.r.t. a unit score scale.

    A stationarity regularizer, not an IRM generalization theorem. Positive
    identities come only from source supervision.
    """
    import mlx.core as mx

    columns = mx.array(positives, dtype=mx.int32)
    positive_scores = scores[columns]
    expectation = mx.sum(mx.softmax(scores) * mx.where(mx.isfinite(scores), scores, 0.))
    positive_expectation = mx.sum(mx.softmax(positive_scores) * positive_scores)
    return (expectation - positive_expectation) ** 2
