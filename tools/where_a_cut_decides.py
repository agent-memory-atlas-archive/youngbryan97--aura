"""At which look on the anchor ladder each cut decides, from samples already paid for.

A full sweep of the 511 bipartitions costs what its cuts cost, and a cut costs 8
anchor-forks when the deciding horizon decides at the first look and 344 when it
never does (the ladder is 8, 16, 32, 64, 96, 128, each read at alpha / 6). Which
end the cuts sit at settles whether the decisive run is hours or days, and the
looks of 29 September kept 128 anchors of samples per cut.

So the ladder can be replayed offline, for no organism and no cortex.

One ladder per cut, not one per horizon. `sweep_cuts_over_lags` collects one set
of rollouts per cut and reads every horizon out of it — `lag_vector` takes a
frame index from the same trajectory — and only the deciding horizon decides
whether a cut draws more anchors. A replay that ran each horizon its own ladder
would report a cost the sweep never pays, and did: it read 344 anchor-forks for a
horizon that in the sweep costs nothing beyond what the deciding one drew.

    tools/where_a_cut_decides.py RUN_DIR [RUN_DIR ...]
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ("context", "intact", "cut", "sham_a", "sham_b")
FULL_CUTS = 511


def _grouped(archive):
    out = {}
    for key in archive.files:
        cut, _, rest = key.partition("__")
        lag, _, arm = rest.partition("__")
        lag = lag if lag.startswith("lag_") else lag.replace("lag", "lag_", 1)
        if arm in ARMS:
            out.setdefault((cut, lag), {})[arm] = archive[key]
    return {k: v for k, v in out.items() if set(ARMS) <= set(v)}


def main(dirs):
    from core.subject.isc_v5 import DECIDING_LAG, LOOKS
    from core.subject.v25_cut import decide_cut

    deciding = f"lag_{DECIDING_LAG}"
    print(f"{"run":>12} {"cut":>13} {"decides at":>11} {"lower":>10} {"forks":>7}")
    spent_per_cut = []
    for run in dirs:
        run = Path(run)
        if not (run / "cut_samples.npz").exists():
            continue
        report = json.loads((run / "subject_core_v25_report.json").read_text())
        plan = (report.get("cuts") or {}).get(deciding)
        if not isinstance(plan, dict) or not plan.get("tau_seconds"):
            continue
        archive = np.load(run / "cut_samples.npz", allow_pickle=True)
        for (cut, lag), arms in sorted(_grouped(archive).items()):
            if lag != deciding:
                continue
            decided_at, lower_at, spent = None, 0.0, 0
            for n in LOOKS:
                if n > len(arms["intact"]):
                    break
                spent += n
                _estimate, lower, _p, _floor = decide_cut(
                    {k: v[:n] for k, v in arms.items()},
                    tau_seconds=float(plan["tau_seconds"]),
                    draws=int(plan.get("draws") or 200),
                    seed=7,
                    alpha=float(plan.get("alpha_per_look") or 0.00833),
                    paired=bool(plan.get("paired")),
                )
                if lower > 0.0:
                    decided_at, lower_at = n, float(lower)
                    break
            name = run.parent.name if run.name.startswith("run_") else run.name
            forks = spent if decided_at else sum(LOOKS)
            spent_per_cut.append(forks)
            print(
                f"{name[:12]:>12} {cut:>13} {str(decided_at or "never"):>11} "
                f"{lower_at:>10.4f} {forks:>7}"
            )
    if spent_per_cut:
        mean = sum(spent_per_cut) / len(spent_per_cut)
        print(
            f"\n  {mean:.0f} anchor-forks per cut over {len(spent_per_cut)} cuts "
            f"(8 decides at once, {sum(LOOKS)} never)"
        )
        print(f"  a full sweep of {FULL_CUTS}: {mean * FULL_CUTS:.0f} anchor-forks")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["."]))
