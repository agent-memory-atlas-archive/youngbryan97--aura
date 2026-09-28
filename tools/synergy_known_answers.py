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

`--drift` asks a different question: what lifts the shifted null on her own
recording, where W,A->D met a bar of 0.40 to 0.50 while every toy so far has met
0.04 to 0.13. It adds columns that only grow, at the share of variance the same
domains carry on her seed-7 recording (`HER_DRIFT_SHARE`). D holds 0.8895 of its
variance in three counters there and is the target of W,A->D and a source of
S,D->C, the two triples whose synergy sits under the bar; W and A carry 0.0044
and 0.0002, so moving the sources alone is the control.

    usage: synergy_known_answers.py [--seeds 3,7,11,19,23]
           synergy_known_answers.py --cycle [--hold 33]
           synergy_known_answers.py --drift target,sources,both
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from core.subject.recording import Recording, slices_from_columns  # noqa: E402
from core.subject.synergy import kraskov_synergy, synergy  # noqa: E402

ROWS = 2400
SEEDS = (3, 7, 11, 19, 23)
KINDS = ("product", "mixed", "additive", "separable", "none")
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


#: Conditions in the cycle a shared schedule runs through, as her campaigns run
#: eight conditions in a fixed order.
CYCLE = 8

#: What share of a domain's variance is columns that only ever grow, measured on
#: her own seed-7 recording at 27dc1dda9 (`--drift` puts these into the toy).
#: D holds nearly all of its variance in three counters, and it is the target of
#: W,A->D and a source of S,D->C, the two triples whose synergy sits under the
#: shifted null. W and A carry almost none, so a drift arm that moves the
#: sources and not the target is the control.
HER_DRIFT_SHARE: dict[str, float] = {"W": 0.0044, "A": 0.0002, "D": 0.8895}


def _ramp(rows: int, width: int, share: float, rng: np.random.Generator) -> np.ndarray:
    """A column that only grows, at `share` of the variance of what it is added to.

    A counter's own shape: a positive step every row, the steps themselves
    varying. Scaled so that the ramp carries `share` of the sum's variance,
    which is the quantity measured on her recording.
    """
    if share <= 0.0:
        return np.zeros((rows, width))
    ramp = np.cumsum(np.abs(rng.normal(size=(rows, width))) + 1.0, axis=0)
    ramp = ramp - ramp.mean(axis=0)
    # var(ramp * k) / (var(ramp * k) + 1) = share, for a unit-variance partner.
    scale = np.sqrt(share / max(1e-9, 1.0 - share)) / np.maximum(ramp.std(axis=0), 1e-12)
    return ramp * scale


def build(kind: str, seed: int, *, cycle: bool = False, hold: int = 1, drift: str = "") -> Recording:
    """One toy recording of the named kind.

    ``cycle`` runs the toy through a shared schedule: eight conditions in a
    fixed order, each moving W, A and what D does by its own offset, a unit
    normal draw per condition. The sources then depend on the target through
    the schedule, as hers do, whatever the coupling between them. ``hold`` is
    how many consecutive rows each condition lasts: one in the first cycle
    table, 33 in her recordings, where a condition holds for a turn's frames.

    ``drift`` adds columns that only grow, at the share of variance the same
    domains carry on her recording (`HER_DRIFT_SHARE`): "target" moves D alone,
    "sources" moves W and A alone, "both" moves all three. Nothing about the
    coupling changes, so a shifted null that rises under drift rises on the
    drift.
    """
    rng = np.random.default_rng(seed)
    w, a = _slow(rng, 3), _slow(rng, 3)
    phase = (np.arange(ROWS) // max(1, int(hold))) % CYCLE
    labels = [f"c{phase[t]}" if cycle else "toy" for t in range(ROWS)]
    if cycle:
        offset_w, offset_a = rng.normal(size=(CYCLE, 3)), rng.normal(size=(CYCLE, 3))
        offset_d = rng.normal(size=(CYCLE, len(PAIRS)))
        w = w + offset_w[phase]
        a = a + offset_a[phase]
    d = np.zeros((ROWS, 8))
    for t in range(1, ROWS):
        push = _drive(kind, w[t - 1], a[t - 1], rng)
        if cycle:
            push = push + offset_d[phase[t]]
        switch = _switching(kind, push, w[t - 1])
        d[t, int(np.argmax(switch + 0.5 * rng.normal(size=len(PAIRS))))] = 1.0
        d[t, 4:] = 0.7 * d[t - 1, 4:] + 0.5 * push + 0.3 * rng.normal(size=len(PAIRS))
    if drift:
        moved = {"target": ("D",), "sources": ("W", "A"), "both": ("W", "A", "D")}[drift]
        if "W" in moved:
            w = w + _ramp(ROWS, 3, HER_DRIFT_SHARE["W"], rng)
        if "A" in moved:
            a = a + _ramp(ROWS, 3, HER_DRIFT_SHARE["A"], rng)
        if "D" in moved:
            # The switch is one-hot and cannot hold a ramp; her D carries its
            # counters in continuous columns, so the toy's do too.
            d[:, 4:] = d[:, 4:] + _ramp(ROWS, 4, HER_DRIFT_SHARE["D"], rng)
    columns = tuple(
        [f"W.w{i}" for i in range(3)] + [f"A.a{i}" for i in range(3)]
        + [f"D.switch{i}" for i in range(4)] + [f"D.level{i}" for i in range(4)]
    )
    return Recording(
        x=np.hstack([w, a, d]), conditions=tuple(labels),
        tags=tuple("ontogeny" for _ in range(ROWS)), times=np.arange(ROWS, dtype=float),
        env=np.zeros((ROWS, 0)), env_names=(), columns=columns,
        slices=slices_from_columns(columns), notes={},
    )


def read(kind: str, seed: int) -> dict[str, object]:
    """One row of the table: the four estimators on one toy, the Kraskov line as the battery reports it."""
    recording = build(kind, seed)
    v3 = synergy(recording, "W", "A", "D", seed=seed, of="change")
    line = kraskov_synergy(recording, "W", "A", "D", seed=seed, draws=KSG_DRAWS, clocks_out=False)
    ksg = line.synergy > line.shift_bar
    fraction = line.synergy / line.joint if line.joint > 1e-9 else 0.0
    return {
        "kind": kind,
        "seed": seed,
        "v3": bool(v3.passes_v3),
        "v3_synergy": float(v3.synergy),
        "v3_bar": float(v3.raw_null_q99),
        "ksg": bool(ksg),
        "ksg_synergy": line.synergy,
        "ksg_bar": line.shift_bar,
        "ksg_fraction": fraction,
        "gain": float(v3.interaction_gain),
        "gain_lower_bound": float(v3.interaction_lower_bound),
        "ksg+gain": bool(ksg and fraction >= 0.10 and v3.interaction_gain > 0.0 and v3.interaction_lower_bound > 0.0),
        "sum_bar": line.additive_bar,
        "ksg-over-sum": bool(line.passes),
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


def read_drift(kind: str, seed: int, drift: str) -> dict[str, object]:
    """One row of the drift table: the Kraskov line's shifted bar under drift."""
    recording = build(kind, seed, drift=drift)
    line = kraskov_synergy(recording, "W", "A", "D", seed=seed, draws=KSG_DRAWS, clocks_out=False)
    v3 = synergy(recording, "W", "A", "D", seed=seed, of="change")
    return {
        "kind": kind,
        "seed": seed,
        "drift": drift,
        "synergy": line.synergy,
        "shift_bar": line.shift_bar,
        "passes": bool(line.passes),
        "v3_synergy": float(v3.synergy),
        "v3_bar": float(v3.raw_null_q99),
        "v3": bool(v3.passes_v3),
    }


def main_drift(seeds: tuple[int, ...], arms: tuple[str, ...]) -> int:
    print("| toy | drift | seed | ksg synergy | shift bar | v3 synergy | v3 shifted bar |")
    print("|---|---|---|---|---|---|---|")
    bars: dict[tuple[str, str], list[float]] = {}
    for drift in arms:
        for kind in CYCLE_KINDS:
            for seed in seeds:
                row = read_drift(kind, seed, drift)
                bars.setdefault((kind, drift), []).append(float(row["shift_bar"]))
                print(
                    f"| {kind} | {drift} | {seed} | {row['synergy']:+.3f} | {row['shift_bar']:+.3f} "
                    f"| {row['v3_synergy']:+.3f} | {row['v3_bar']:+.3f} |"
                )
    print()
    for (kind, drift), values in sorted(bars.items()):
        reached = sum(1 for value in values if value >= 0.40)
        print(
            f"{kind} under {drift}: shift bar median {float(np.median(values)):+.3f}, "
            f"{reached} of {len(values)} seeds at or above her 0.40"
        )
    return 0


def read_cycle(kind: str, seed: int, hold: int = 1) -> dict[str, object]:
    """One row of the cycle table: the Kraskov line under each first null, on a toy with a shared schedule."""
    recording = build(kind, seed, cycle=True, hold=hold)
    row: dict[str, object] = {"kind": kind, "seed": seed}
    for first_null in ("shift", "condition"):
        line = kraskov_synergy(
            recording, "W", "A", "D", seed=seed, draws=KSG_DRAWS, clocks_out=False, first_null=first_null
        )
        row[first_null] = bool(line.passes)
        row[f"{first_null}_bar"] = line.shift_bar
        row["synergy"] = line.synergy
        row["additive_bar"] = line.additive_bar
    return row


def main_cycle(seeds: tuple[int, ...], hold: int = 1) -> int:
    counts = {name: dict.fromkeys(CYCLE_KINDS, 0) for name in ("shift", "condition")}
    for kind in CYCLE_KINDS:
        for seed in seeds:
            row = read_cycle(kind, seed, hold)
            for name in counts:
                counts[name][kind] += int(row[name])
            print(
                f"cycle {kind:9s} seed {seed:2d}: syn {row['synergy']:+.4f} | shift bar {row['shift_bar']:+.4f} "
                f"passes {row['shift']!s:5s} | condition bar {row['condition_bar']:+.4f} passes {row['condition']!s:5s} "
                f"| additive bar {row['additive_bar']:+.4f}",
                flush=True,
            )
    need = len(seeds) - 1
    for name, row in counts.items():
        ok = row["product"] >= need and row["separable"] <= 1 and row["none"] <= 1
        print(f"{name}: {row} -> {'qualifies' if ok else 'does not qualify'}")
    return 0


#: The toys read with a shared schedule.
CYCLE_KINDS = ("product", "separable", "none")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--kinds", default=",".join(KINDS))
    parser.add_argument("--cycle", action="store_true", help="the toys with a shared condition cycle, both first nulls")
    parser.add_argument("--hold", type=int, default=1, help="rows each condition of the cycle lasts")
    parser.add_argument(
        "--drift",
        default="",
        help="columns that only grow, at her own shares: target, sources, both, "
        "or several separated by commas",
    )
    args = parser.parse_args()
    if args.drift:
        return main_drift(
            tuple(int(s) for s in args.seeds.split(",")),
            tuple(name.strip() for name in args.drift.split(",") if name.strip()),
        )
    if args.cycle:
        return main_cycle(tuple(int(s) for s in args.seeds.split(",")), args.hold)
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
