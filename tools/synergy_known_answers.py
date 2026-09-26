#!/usr/bin/env python3
"""Can a synergy estimator see a product that is known to be there?

The study in docs/SYNERGY_KNOWN_ANSWERS.md. Four toy systems at five seeds,
2,400 turn rows each, the length of a 300-round campaign. W and A are three
slow AR(1) channels each; the target D holds a four-state switch and four
continuous columns, driven by

    product   W_i * A_j
    mixed     (W_i - A_j) + W_i * A_j
    additive  (W_i + A_j) / 2, the switch taking the largest of those sums
    separable (W_i + A_j) / 2, the switch taking the largest W_i alone
    none      nothing either source does

and each is read three ways, W,A -> D on the change:

    v3        the battery's own line, `SynergyReport.passes_v3`
    ksg       MMI synergy through a Kraskov estimator (Kraskov, Stoegbauer and
              Grassberger 2004, estimator 1, max norm, k = 3) on the same
              copula-normal components, against the same shifted null
    ksg+gain  ksg, a Kraskov fraction of at least 0.10, and the v3 line's
              held-out interaction gain positive with a positive lower bound
    ksg-over-sum
              ksg, and the Kraskov synergy above what surrogates holding only
              the additive part of the target show: the best additive fit plus
              its residual permuted over rows

An estimator qualifies when it passes product and mixed at four of five seeds
and passes separable and none at no more than one.
    usage: synergy_known_answers.py [--seeds 3,7,11,19,23]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.subject.recording import Recording, slices_from_columns  # noqa: E402
from core.subject.synergy import _components, _copula_normal, synergy  # noqa: E402

ROWS = 2400
SEEDS = (3, 7, 11, 19, 23)
KINDS = ("product", "mixed", "additive", "separable", "none")
#: Neighbours for the Kraskov estimator, the middle of the 2 to 4 its authors
#: recommend.
K = 3
#: Null draws for the Kraskov line. A Kraskov MI costs a few hundred times a
#: Gaussian one; 200 draws still put two past the 0.99 quantile. The v3 line
#: keeps its own 1,000.
KSG_DRAWS = 200
#: Which W and A channels drive each of D's four columns.
PAIRS = ((0, 0), (1, 1), (2, 2), (0, 2))


def _slow(rng: np.random.Generator, width: int, rho: float = 0.9) -> np.ndarray:
    x = np.zeros((ROWS, width))
    for t in range(1, ROWS):
        x[t] = rho * x[t - 1] + rng.normal(size=width) * np.sqrt(1 - rho**2)
    return x


def _drive(kind: str, w: np.ndarray, a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if kind == "product":
        return np.array([w[i] * a[j] for i, j in PAIRS])
    if kind == "mixed":
        return np.array([(w[i] - a[j]) + w[i] * a[j] for i, j in PAIRS])
    if kind in {"additive", "separable"}:
        return np.array([(w[i] + a[j]) / 2 for i, j in PAIRS])
    return rng.normal(size=len(PAIRS))


def _switching(kind: str, push: np.ndarray, w: np.ndarray) -> np.ndarray:
    """What the switch picks the largest of.

    In `additive` it is the sums themselves, and which of four sums is largest
    is not a sum: the switch carries an interaction. `separable` hands the
    switch W alone, so nothing D does depends on W and A together.
    """
    if kind == "separable":
        return np.array([w[i] for i, _ in PAIRS])
    return push


def build(kind: str, seed: int) -> Recording:
    """One toy recording of the named kind."""
    rng = np.random.default_rng(seed)
    w, a = _slow(rng, 3), _slow(rng, 3)
    d = np.zeros((ROWS, 8))
    for t in range(1, ROWS):
        push = _drive(kind, w[t - 1], a[t - 1], rng)
        switch = _switching(kind, push, w[t - 1])
        d[t, int(np.argmax(switch + 0.5 * rng.normal(size=len(PAIRS))))] = 1.0
        d[t, 4:] = 0.7 * d[t - 1, 4:] + 0.5 * push + 0.3 * rng.normal(size=len(PAIRS))
    columns = tuple(
        [f"W.w{i}" for i in range(3)] + [f"A.a{i}" for i in range(3)]
        + [f"D.switch{i}" for i in range(4)] + [f"D.level{i}" for i in range(4)]
    )
    return Recording(
        x=np.hstack([w, a, d]), conditions=tuple("toy" for _ in range(ROWS)),
        tags=tuple("ontogeny" for _ in range(ROWS)), times=np.arange(ROWS, dtype=float),
        env=np.zeros((ROWS, 0)), env_names=(), columns=columns,
        slices=slices_from_columns(columns), notes={},
    )


def ksg_mi(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """Kraskov estimator 1, max norm, in nats."""
    x = x + 1e-10 * rng.normal(size=x.shape)
    y = y + 1e-10 * rng.normal(size=y.shape)
    joint = np.hstack([x, y])
    eps = cKDTree(joint).query(joint, k=K + 1, p=np.inf)[0][:, -1]
    nx = cKDTree(x).query_ball_point(x, eps - 1e-15, p=np.inf, return_length=True) - 1
    ny = cKDTree(y).query_ball_point(y, eps - 1e-15, p=np.inf, return_length=True) - 1
    n = x.shape[0]
    return float(digamma(K) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))


def _ksg_synergy(a: np.ndarray, b: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    joint = ksg_mi(np.hstack([a, b]), y, rng)
    return joint - max(ksg_mi(a, y, rng), ksg_mi(b, y, rng)), joint


#: Degrees the additive model may take for each source; the one with the
#: lowest held-out loss is used.
ADDITIVE_DEGREES = (1, 2, 3)
#: Folds the additive model's degree is chosen over.
ADDITIVE_FOLDS = 5


def _additive_basis(a: np.ndarray, b: np.ndarray, degree: int) -> np.ndarray:
    """Powers of each source's columns up to `degree`, and no product between sources."""
    columns = [np.ones((a.shape[0], 1))]
    for source in (a, b):
        for power in range(1, degree + 1):
            columns.append(source**power)
    return np.hstack(columns)


def additive_fit(a: np.ndarray, b: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The best additive account of y from a and b, its degree chosen by held-out loss."""
    rows = y.shape[0]
    fold = np.arange(rows) * ADDITIVE_FOLDS // rows
    best, best_loss = ADDITIVE_DEGREES[0], np.inf
    for degree in ADDITIVE_DEGREES:
        basis = _additive_basis(a, b, degree)
        loss = 0.0
        for k in range(ADDITIVE_FOLDS):
            train, test = fold != k, fold == k
            coef = np.linalg.lstsq(basis[train], y[train], rcond=None)[0]
            loss += float(np.sum((y[test] - basis[test] @ coef) ** 2))
        if loss < best_loss:
            best, best_loss = degree, loss
    basis = _additive_basis(a, b, best)
    return basis @ np.linalg.lstsq(basis, y, rcond=None)[0]


def additive_null(a: np.ndarray, b: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> float:
    """The 0.99 quantile of the synergy a target with only the additive part of y would show.

    Each surrogate is the additive fit plus its own residual permuted over rows,
    which keeps every additive dependence on the sources and the noise level,
    and removes anything the sources do together.
    """
    fitted = additive_fit(a, b, y)
    residual = y - fitted
    draws = []
    for _ in range(KSG_DRAWS):
        surrogate = fitted + residual[rng.permutation(residual.shape[0])]
        draws.append(_ksg_synergy(a, b, surrogate, rng)[0])
    return float(np.quantile(draws, 0.99))


def read(kind: str, seed: int) -> dict[str, object]:
    """One row of the table: the three estimators on one toy."""
    recording = build(kind, seed)
    v3 = synergy(recording, "W", "A", "D", seed=seed, of="change")
    following = recording.domain("D")
    a, b, y = (
        _copula_normal(_components(block))
        for block in (recording.domain("W")[:-1], recording.domain("A")[:-1], following[1:] - following[:-1])
    )
    rng = np.random.default_rng(seed)
    value, joint = _ksg_synergy(a, b, y, rng)
    rows = a.shape[0]
    nulls = []
    for _ in range(KSG_DRAWS):
        shift = int(rng.integers(rows // 8, rows - rows // 8))
        nulls.append(_ksg_synergy(np.roll(a, shift, 0), np.roll(b, shift, 0), y, rng)[0])
    bar = float(np.quantile(nulls, 0.99))
    ksg = value > bar
    fraction = value / joint if joint > 1e-9 else 0.0
    sum_bar = additive_null(a, b, y, rng)
    return {
        "kind": kind,
        "seed": seed,
        "v3": bool(v3.passes_v3),
        "v3_synergy": float(v3.synergy),
        "v3_bar": float(v3.raw_null_q99),
        "ksg": bool(ksg),
        "ksg_synergy": value,
        "ksg_bar": bar,
        "ksg_fraction": fraction,
        "gain": float(v3.interaction_gain),
        "gain_lower_bound": float(v3.interaction_lower_bound),
        "ksg+gain": bool(ksg and fraction >= 0.10 and v3.interaction_gain > 0.0 and v3.interaction_lower_bound > 0.0),
        "sum_bar": sum_bar,
        "ksg-over-sum": bool(ksg and value > sum_bar),
    }


def qualifies(counts: dict[str, int], seeds: int) -> bool:
    """Product and mixed at four of five; the separable sum and none at no more than one.

    The rule was written with `additive` as the sum that must fail. Its switch
    takes the largest of four sums, which is an interaction, so it is reported
    and no longer counted either way; `separable` is the sum that must fail.
    """
    need = seeds - 1
    return (
        counts["product"] >= need
        and counts["mixed"] >= need
        and counts.get("separable", 0) <= 1
        and counts["none"] <= 1
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--kinds", default=",".join(KINDS))
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    kinds = tuple(args.kinds.split(","))
    counts = {name: dict.fromkeys(KINDS, 0) for name in ("v3", "ksg", "ksg+gain", "ksg-over-sum")}
    for kind in kinds:
        for seed in seeds:
            row = read(kind, seed)
            for name in counts:
                counts[name][kind] += int(row[name])
            print(
                f"{kind:9s} seed {seed:2d}: v3 {row['v3']!s:5s} syn {row['v3_synergy']:+.4f} "
                f"bar {row['v3_bar']:+.4f} | ksg {row['ksg']!s:5s} syn {row['ksg_synergy']:+.4f} "
                f"bar {row['ksg_bar']:+.4f} | gain {row['gain']:+.4f} lb {row['gain_lower_bound']:+.4f} "
                f"| ksg+gain {row['ksg+gain']!s} | sum bar {row['sum_bar']:+.4f} "
                f"ksg-over-sum {row['ksg-over-sum']!s}",
                flush=True,
            )
    for name, row in counts.items():
        print(f"{name}: {row} -> {'qualifies' if qualifies(row, len(seeds)) else 'does not qualify'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
