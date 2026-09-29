"""Whether two subject runs recorded the same organism, and where they part if not.

A sharded sweep is a sweep of her only if every shard is the same organism. Two
runs on one seed with `--baseline-only` should record the same baseline to the
bit; this reads both recordings and says so, or names the first frame they
differ at and the columns that differ there.

    python tools/same_organism.py RUN_A RUN_B

Exit 0 when the recordings are identical, 1 when they are not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _recording(run: Path) -> tuple[np.ndarray, list[str]]:
    found = run / "core_state.npz" if (run / "core_state.npz").exists() else sorted(run.glob("run_*/core_state.npz"))[-1]
    manifest = json.loads((found.parent / "core_state_manifest.json").read_text(encoding="utf-8"))
    return np.load(found)["x"], list(manifest["columns"])


def compare(a: Path, b: Path, *, show: int = 12) -> dict:
    x, columns = _recording(a)
    y, other = _recording(b)
    if columns != other:
        return {"same": False, "why": "the two runs record different columns"}
    frames = min(len(x), len(y))
    apart = x[:frames] != y[:frames]
    rows = np.flatnonzero(apart.any(axis=1))
    report: dict = {"frames": [len(x), len(y)], "same": len(x) == len(y) and rows.size == 0}
    if rows.size:
        first = int(rows[0])
        where = np.flatnonzero(apart[first])
        report.update(
            first_frame_apart=first,
            columns_apart_there=int(where.size),
            columns=[columns[i] for i in where[:show]],
            largest_gap=float(np.max(np.abs(x[first] - y[first]))),
            frames_apart=int(rows.size),
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    report = compare(args.a, args.b)
    print(json.dumps(report, indent=2))
    return 0 if report.get("same") else 1


if __name__ == "__main__":
    sys.exit(main())
