"""Whether the cut costs prediction within a stretch of her life, and against a null.

`phi_do` fits on the past and tests on the future across five forward-chaining
folds, and reads the mean of them with a standard error. That design assumes the
system is the same system in every fold. She is learning — weights, visit
histories and ledgers all fill up as a run goes on — so a model fitted on early
rows and tested on late ones is fitted on a different organism.

On `whole-s7-27dc1dda9` the forward-chaining folds were
`[-0.239, -0.0, 0.0, 0.342, 0.353]`: a point estimate of 0.0912, well past the
0.05 bar, with a lower bound of -0.131 that one fold owns. Scored instead on each
contiguous fifth at the same cut, every fifth is positive with a positive lower
bound, and every fifth of a row-shuffled copy is zero.

    tools/audit_stationary_irreducibility.py <run_NNN> [--cut PIM] [--parts 5]

It reports and decides nothing. A positive reading here is evidence about the
sign of the coupling and not about its size: the within-stretch values are two to
nine thousandths, and the preregistered bar is five hundredths.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

#: Rows below which a stretch cannot carry the estimator's own folds. `phi_do`
#: needs enough rows per fold to fit at all; below this the reading is about the
#: stretch being short rather than about her.
MIN_ROWS_PER_PART: int = 2_000


def stretches(recording: Any, parts: int) -> list[tuple[int, int]]:
    """Contiguous equal stretches, as row bounds."""
    total = int(recording.frames)
    edges = [int(total * index / parts) for index in range(parts + 1)]
    return [(edges[index], edges[index + 1]) for index in range(parts)]


def rows_of(recording: Any, index: np.ndarray) -> Any:
    """The recording restricted to these rows, everything else carried along."""
    return replace(
        recording,
        x=recording.x[index],
        conditions=tuple(np.asarray(recording.conditions)[index]),
        tags=tuple(np.asarray(recording.tags)[index]),
        times=recording.times[index],
        env=recording.env[index],
    )


def shuffled(recording: Any, *, seed: int) -> Any:
    """The same rows in a random order: every marginal kept, every relation gone."""
    order = np.random.default_rng(seed).permutation(int(recording.frames))
    out = rows_of(recording, order)
    # Times stay as they were. A shuffled recording is not a recording of a life
    # and its clock is not evidence about one; leaving the timestamps in place
    # keeps any consumer that reads them from inferring an order from the null.
    return replace(out, times=recording.times)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="a run_NNN directory")
    parser.add_argument(
        "--cut",
        default="",
        help="one side of the bipartition as domain letters, e.g. PIM. "
        "Default: the cheapest cut of the whole recording.",
    )
    parser.add_argument("--parts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.subject.irreducibility import phi_do
    from core.subject.recording import load_recording
    from core.subject.state import DOMAINS

    recording = load_recording(args.run)
    whole = phi_do(recording).as_dict()
    if args.cut:
        side = tuple(letter for letter in args.cut.upper() if letter in DOMAINS)
    else:
        side = tuple(whole["best_cut"][0])
    other = tuple(name for name in DOMAINS if name not in side)
    if not side or not other:
        raise SystemExit(f"a cut needs both sides; got {side!r} against {other!r}")
    cut = (side, other)

    print(f"{recording.frames} frames, {recording.x.shape[1]} columns")
    print(
        f"whole run, forward-chaining folds: phi {whole['phi_do']:+.6f} "
        f"lower {whole['lower_bound']:+.6f} at {''.join(side)}|{''.join(other)}"
    )
    print(f"  folds {[round(value, 6) for value in whole['held_out']]}")

    fake = shuffled(recording, seed=args.seed)
    rows: list[dict[str, Any]] = []
    print(f"\n{'stretch':>8}  {'as recorded':>28}  {'row-shuffled':>28}")
    for index, (low, high) in enumerate(stretches(recording, args.parts), start=1):
        span = np.arange(low, high)
        if span.size < MIN_ROWS_PER_PART:
            print(f"{index:>8}  {span.size} rows is too few to fold; skipped")
            continue
        real = phi_do(rows_of(recording, span), at=cut).as_dict()
        null = phi_do(rows_of(fake, span), at=cut).as_dict()
        rows.append(
            {
                "stretch": index,
                "rows": int(span.size),
                "phi": real["phi_do"],
                "lower_bound": real["lower_bound"],
                "null_phi": null["phi_do"],
                "null_lower_bound": null["lower_bound"],
            }
        )
        print(
            f"{index:>8}  phi {real['phi_do']:+.6f} lower {real['lower_bound']:+.6f}"
            f"    phi {null['phi_do']:+.6f} lower {null['lower_bound']:+.6f}"
        )

    held = sum(1 for row in rows if row["lower_bound"] > 0.0)
    null_held = sum(1 for row in rows if row["null_lower_bound"] > 0.0)
    print(
        f"\n{held} of {len(rows)} stretches cost prediction with a positive lower "
        f"bound; {null_held} of {len(rows)} do on shuffled rows."
    )
    if rows:
        print(
            "largest within-stretch reading "
            f"{max(row['phi'] for row in rows):.6f}, against a preregistered bar of 0.05"
        )
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "cut": ["".join(side), "".join(other)],
                    "whole": {k: whole[k] for k in ("phi_do", "lower_bound", "held_out")},
                    "stretches": rows,
                },
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
