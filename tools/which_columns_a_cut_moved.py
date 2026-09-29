"""Which of her columns a cut moved, read from the samples a look kept.

A cut's verdict is one number: how far the cut arm's future lies from the
untouched arm's, less how far two untouched arms lie apart. When a cut stays
undecided the number says it moved too little and not where. Each anchor forks
both arms from one snapshot, so the per-anchor difference between them is the
cut's own effect on each column, and the sham pair gives the same difference
with nothing cut.

    python tools/which_columns_a_cut_moved.py RUN_DIR [--top 15]

RUN_DIR holds cut_samples.npz (written by `--keep-cut-samples`) and the
recording's core_state_manifest.json, which names the columns. Each column's
move is the mean absolute paired difference in units of that column's spread
across the anchors' untouched futures, so a column that never moves reads zero
and a clock that moves the same in both arms reads zero too.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _columns(run: Path) -> list[str]:
    manifest = json.loads((run / "core_state_manifest.json").read_text(encoding="utf-8"))
    return list(manifest["columns"])


def moved(run: Path) -> dict[str, dict[int, dict[str, object]]]:
    """For each cut and horizon: per-column moves under the cut and under the sham, by domain."""
    samples = np.load(run / "cut_samples.npz")
    columns = _columns(run)
    found: dict[tuple[str, int], dict[str, np.ndarray]] = defaultdict(dict)
    for key in samples.files:
        match = re.fullmatch(r"(.+)__lag(\d+)__(\w+)", key)
        if match:
            found[(match.group(1), int(match.group(2)))][match.group(3)] = samples[key]
    out: dict[str, dict[int, dict[str, object]]] = defaultdict(dict)
    for (cut, lag), slot in sorted(found.items()):
        intact, cut_arm = slot["intact"], slot["cut"]
        sham_a, sham_b = slot["sham_a"], slot["sham_b"]
        width = intact.shape[1]
        names = columns[-width:] if len(columns) >= width else [f"col{i}" for i in range(width)]
        spread = intact.std(axis=0)
        unit = np.where(spread > 1e-12, spread, np.inf)
        by_cut = np.mean(np.abs(cut_arm - intact), axis=0) / unit
        by_sham = np.mean(np.abs(sham_b - sham_a), axis=0) / unit
        domains: dict[str, float] = defaultdict(float)
        for name, value in zip(names, by_cut, strict=True):
            domains[name.split(".", 1)[0]] += float(value)
        order = np.argsort(-by_cut)
        out[cut][lag] = {
            "anchors": int(intact.shape[0]),
            "by_domain": dict(sorted(domains.items(), key=lambda item: -item[1])),
            "top": [(names[i], round(float(by_cut[i]), 4), round(float(by_sham[i]), 4)) for i in order],
            "moved_columns": int(np.sum(by_cut > by_sham)),
            "total": round(float(np.sum(by_cut)), 4),
            "sham_total": round(float(np.sum(by_sham)), 4),
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    report = moved(args.run)
    for cut, by_lag in report.items():
        for lag, reading in by_lag.items():
            print(f"{cut} at lag {lag}, {reading['anchors']} anchors: total move {reading['total']} "
                  f"(sham {reading['sham_total']}), {reading['moved_columns']} columns above their sham")
            print("  by domain: " + ", ".join(f"{d} {v:.3f}" for d, v in reading["by_domain"].items()))
            for name, value, sham in reading["top"][: args.top]:
                print(f"    {name:40s} {value:8.4f}   sham {sham:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
