#!/usr/bin/env python3
"""Decide a look's cuts twice on its own kept samples: standardised and whitened.

The preregistration of 28 September, night, names `--whiten` and says what it is
held to: "The next carrier run reads both, on the same samples, and reports
both. If the whitened rate does not clear the drift tolerance the fix has failed
and the blocker stands."

A look run with `--keep-cut-samples` has already paid for those samples. This
reads them, so the second estimator costs no organism and no cortex, and the two
readings are of the same rows rather than of two runs.

Nothing here moves a bar. It reports the excess rate, its lower bound and the
decision under each metric, and each cut's drift under an invertible
re-encoding, which is the blocker the whitened metric exists to clear.

    tools/read_a_look_both_ways.py <run_dir> [--out both_ways.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ARMS = ("context", "intact", "cut", "sham_a", "sham_b")


def _looks(archive: Any) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """The kept arrays, grouped by (cut, lag)."""
    grouped: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key in archive.files:
        cut, _, rest = key.partition("__")
        lag, _, arm = rest.partition("__")
        # The arrays are keyed `lag33`, the report's design `lag_33`.
        lag = lag if lag.startswith("lag_") else lag.replace("lag", "lag_", 1)
        if arm in ARMS:
            grouped.setdefault((cut, lag), {})[arm] = archive[key]
    return {where: arms for where, arms in grouped.items() if set(ARMS) <= set(arms)}


def _design(run_dir: Path) -> dict[str, Any]:
    report = json.loads((run_dir / "subject_core_v25_report.json").read_text())
    cuts = report.get("cuts", {})
    return {
        lag: {
            "tau_seconds": float(detail.get("tau_seconds") or 0.0),
            "alpha": float(detail.get("alpha_per_look") or 0.05),
            "draws": int(detail.get("draws") or 200),
            "paired": bool(detail.get("paired")),
            "verdicts": {v["cut"]: v for v in detail.get("verdicts", [])},
        }
        for lag, detail in cuts.items()
        if isinstance(detail, dict) and detail.get("tau_seconds")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out", default="both_ways.json")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    from core.subject.v25_cut import decide_cut

    samples_file = args.run_dir / "cut_samples.npz"
    if not samples_file.exists():
        print(f"no kept samples in {args.run_dir}; the look needed --keep-cut-samples")
        return 1
    archive = np.load(samples_file, allow_pickle=True)
    design = _design(args.run_dir)
    rows: list[dict[str, Any]] = []
    for (cut, lag), arms in sorted(_looks(archive).items()):
        plan = design.get(lag)
        if plan is None:
            continue
        row: dict[str, Any] = {"cut": cut, "lag": lag, "anchors": int(len(arms["intact"]))}
        for metric, whiten in (("standardised", False), ("whitened", True)):
            estimate, lower, _p, _floor = decide_cut(
                arms,
                tau_seconds=plan["tau_seconds"],
                draws=plan["draws"],
                seed=args.seed + 5,
                alpha=plan["alpha"],
                whiten=whiten,
                paired=plan["paired"],
            )
            row[metric] = {
                "excess_rate": round(float(estimate.excess_rate), 6),
                "lower_bound": round(float(lower), 6),
                "decided": bool(lower > 0.0),
            }
        rows.append(row)
        print(
            f"{cut:>12} {lag}  standardised excess={row['standardised']['excess_rate']:+.6f} "
            f"lower={row['standardised']['lower_bound']:+.6f} decided={row['standardised']['decided']}"
        )
        print(
            f"{'':>12} {' ' * len(lag)}  whitened     excess={row['whitened']['excess_rate']:+.6f} "
            f"lower={row['whitened']['lower_bound']:+.6f} decided={row['whitened']['decided']}"
        )
    out = args.run_dir / args.out
    out.write_text(json.dumps({"run": str(args.run_dir), "cuts": rows}, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
