"""Bounded exact discourse inference from supplied relational factors.

Forward filtering and retrospective smoothing use the same finite model.
The probabilities are conditional on those factors, not calibrated claims of
semantic correctness. Evidence later than the cutoff never enters either.
"""

from dataclasses import dataclass, replace
from itertools import product
from types import MappingProxyType

import numpy as np
from scipy.special import logsumexp

from core.learning.semantic_semasiographic import source_key


@dataclass(frozen=True)
class DiscourseFactor:
    positions: tuple[int, ...]
    log_weights: object
    source: str
    available_at: int

    def __post_init__(self):
        positions = tuple(self.positions)
        values = np.array(self.log_weights, dtype=float, copy=True)
        if (not positions or len(set(positions)) != len(positions)
                or any(type(p) is not int or p < 0 for p in positions)
                or values.ndim != len(positions) or np.isnan(values).any() or np.isposinf(values).any()
                or not np.isfinite(values).any() or not self.source
                or type(self.available_at) is not int or self.available_at < 0):
            raise ValueError("invalid source-identified discourse factor")
        values.setflags(write=False)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "log_weights", values)


@dataclass(frozen=True)
class DiscoursePosterior:
    marginals: tuple
    log_marginals: tuple
    evidence: tuple[str, ...]
    excluded: tuple[str, ...]
    cutoff: int
    mode: str

    def binding_costs(self, role_positions):
        """Pass measured-model marginals to the existing binder's evidence API."""
        if any(type(p) is not int or not 0 <= p < len(self.marginals) for p in role_positions.values()):
            raise ValueError("discourse role positions differ from posterior")
        return {(role, key): -log_probability
                for role, p in role_positions.items() for key, log_probability in self.log_marginals[p].items()
                if np.isfinite(log_probability)}


def infer_discourse_references(candidates, factors, *, cutoff, mode="smooth", max_paths=65_536):
    """Enumerate bounded assignments using SciPy's stable sum-product algebra."""
    candidates, factors = tuple(tuple(c) for c in candidates), tuple(factors)
    if (not candidates or len(candidates) > 32 or any(not c or len(set(c)) != len(c) for c in candidates)
            or type(cutoff) is not int or cutoff < 0 or mode not in {"filter", "smooth"}
            or type(max_paths) is not int or not 0 < max_paths <= 65_536 or len(factors) > 256):
        raise ValueError("invalid bounded discourse model")
    for c in candidates:
        for key in c:
            source_key(key)
    for factor in factors:
        if (not isinstance(factor, DiscourseFactor) or max(factor.positions) >= len(candidates)
                or factor.log_weights.shape != tuple(len(candidates[p]) for p in factor.positions)):
            raise ValueError("discourse factors differ from source candidate identities")
    count = 1
    for c in candidates:
        count *= len(c)
        if count > max_paths:
            raise ValueError("discourse assignment space exceeds its declared exact bound")
    admitted = tuple(f for f in factors if f.available_at <= cutoff)
    marginals, logs = [], []
    full_paths, full_weights, full_normalizer = None, None, None
    for target in range(len(candidates)):
        relevant = admitted if mode == "smooth" else tuple(f for f in admitted if max(f.positions) <= target)
        stop = len(candidates) if mode == "smooth" else target + 1
        if mode == "smooth" and full_paths is not None:
            paths, weights, normalizer = full_paths, full_weights, full_normalizer
        else:
            paths = np.asarray(tuple(product(*(range(len(c)) for c in candidates[:stop]))), dtype=np.intp)
            weights = np.zeros(len(paths))
            for factor in relevant:
                weights += factor.log_weights[tuple(paths[:, p] for p in factor.positions)]
            normalizer = logsumexp(weights)
            if mode == "smooth":
                full_paths, full_weights, full_normalizer = paths, weights, normalizer
        if not np.isfinite(normalizer):
            raise ValueError("discourse premises exclude every assignment")
        log_row = {key: float(logsumexp(weights[paths[:, target] == index]) - normalizer)
                   for index, key in enumerate(candidates[target])}
        row = {key: float(np.exp(value)) for key, value in log_row.items()}
        marginals.append(MappingProxyType(row))
        logs.append(MappingProxyType(log_row))
    return DiscoursePosterior(tuple(marginals), tuple(logs), tuple(sorted({f.source for f in admitted})),
        tuple(sorted({f.source for f in factors if f.available_at > cutoff})), cutoff, mode)


def resolve_discourse_binding(engine, evidence, candidates, factors, role_positions, *, cutoff,
                             mode="smooth", weight=1., baseline_costs=None, **options):
    """Combine available discourse support with native evidence in one joint solve.

    A zero conditional probability excludes a source only under the supplied
    factor model. Numerical underflow does not create such a hard exclusion.
    """
    from core.learning.semantic_context_binding import bind_context_roles

    if not np.isfinite(weight) or weight < 0:
        raise ValueError("discourse evidence weight must be finite and nonnegative")
    posterior = infer_discourse_references(candidates, factors, cutoff=cutoff, mode=mode)
    additions = posterior.binding_costs(role_positions)
    if not set(role_positions) <= {r.identity for r in evidence.roles}:
        raise ValueError("discourse references have no corresponding grounded roles")
    if any(r.tick > cutoff for r in evidence.roles):
        raise ValueError("grounded reference clock exceeds the discourse evidence cutoff")
    costs = engine.costs(evidence, baseline_costs=baseline_costs)
    roles = []
    keys = {r.key for r in evidence.context.referents}
    for role in evidence.roles:
        if role.identity not in role_positions:
            roles.append(role)
            continue
        log_row = posterior.log_marginals[role_positions[role.identity]]
        eligible = {r.key for r in evidence.context.referents if evidence.context.eligible(role, r)}
        if not set(log_row) <= keys or not eligible <= set(log_row):
            raise ValueError("discourse evidence does not cover the eligible source identities")
        supported = tuple(sorted(k for k in eligible if np.isfinite(log_row[k])))
        roles.append(replace(role, referents=supported) if weight else role)
    conditioned = replace(evidence, roles=tuple(roles))
    by_role = {r.identity: r for r in roles}
    by_source = {r.key: r for r in evidence.context.referents}
    costs = {key: value + weight * additions.get(key, 0.) for key, value in costs.items()
             if evidence.context.eligible(by_role[key[0]], by_source[key[1]])}
    result = bind_context_roles(conditioned.context, conditioned.roles, costs=costs,
                                relations=conditioned.relations, **options)
    return result, {"schema": "aura.discourse_binding.v1", "source_id": evidence.source_id,
        "cutoff": cutoff, "mode": mode, "weight": weight, "evidence": posterior.evidence,
        "excluded": posterior.excluded, "posterior_is_calibrated_accuracy": False,
        "costs": [(role, key, value) for (role, key), value in sorted(costs.items())],
        "status": result.status, "bindings": result.bindings, "margin": result.margin,
        "serving_authority": False}
