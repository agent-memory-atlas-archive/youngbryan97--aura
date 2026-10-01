"""Exact linear-surrogate capacity diagnostics, not a transformer ceiling."""

import numpy as np


def reduced_rank_binding_fit(features, targets, *, rank, baseline=None, rcond=None):
    """Minimize ||X W - (Y - baseline)||_F over rank(W) <= rank.

    Truncating the unwhitened coefficient SVD is not prediction-optimal for
    anisotropic X. Work in X's observed column space, then unwhiten. The
    reported lower bound applies only to this declared linear surrogate and
    these observations, not to Aura's nonlinear adapted decoder blocks.
    """
    features, targets = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    if (features.ndim != 2 or targets.ndim != 2 or min(features.shape) < 1
            or targets.shape[0] != features.shape[0] or targets.shape[1] < 1
            or type(rank) is not int or rank < 1
            or not np.isfinite(features).all() or not np.isfinite(targets).all()):
        raise ValueError("binding capacity needs finite aligned multi-output observations")
    baseline = np.zeros_like(targets) if baseline is None else np.asarray(baseline, dtype=np.float64)
    if baseline.shape != targets.shape or not np.isfinite(baseline).all():
        raise ValueError("binding capacity baseline observations differ")
    residual = targets - baseline
    left, singular, right = np.linalg.svd(features, full_matrices=False)
    rcond = max(features.shape) * np.finfo(float).eps if rcond is None else float(rcond)
    if not np.isfinite(rcond) or not 0 <= rcond < 1:
        raise ValueError("binding capacity numerical rank threshold differs")
    observed = singular > rcond * singular[0]
    left, singular, right = left[:, observed], singular[observed], right[observed]
    coordinates = left.T @ residual
    inaccessible = residual - left @ coordinates
    u, spectrum, v = np.linalg.svd(coordinates, full_matrices=False)
    retained = min(rank, len(spectrum))
    approximation = (u[:, :retained] * spectrum[:retained]) @ v[:retained]
    coefficient = (right.T / singular) @ approximation if singular.size else np.zeros((features.shape[1], targets.shape[1]))
    irreducible = float(np.sum(inaccessible ** 2))
    tail = float(np.sum(spectrum[retained:] ** 2))
    return coefficient, {
        "schema": "aura.binding_linear_capacity.v1", "scope": "observed_linear_surrogate_only",
        "requested_rank": rank, "observed_input_rank": int(singular.size),
        "retained_rank": retained, "inaccessible_error": irreducible,
        "rank_tail_error": tail, "minimum_squared_error": irreducible + tail,
        "actual_squared_error": float(np.sum((features @ coefficient - residual) ** 2)),
        "spectrum": spectrum.tolist(), "native_decoder_ceiling": False,
    }
