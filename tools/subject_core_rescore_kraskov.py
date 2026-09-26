#!/usr/bin/env python3
"""The Kraskov synergy line for a campaign that finished before it was recorded.

    python tools/subject_core_rescore_kraskov.py RUN_DIR [...]

Scores the four declared triples with the line in docs/SYNERGY_KNOWN_ANSWERS.md
(`core.subject.synergy.kraskov_suite`) on the run's own saved recording, at the
run's seed, with the code as committed. Before it scores anything it recomputes
the run's v3 reading on the change with the counters out and refuses the run if
that does not match what the run recorded, so a recording that has drifted from
its report is not read.

It never touches the run's report. It writes `kraskov_synergy.json` beside it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SIDECAR = "kraskov_synergy.json"
#: A report rounds a v3 synergy to four places.
ROUNDING = 1e-4


def _commit() -> str:
    try:
        return subprocess.run(
            ["git", "-c", "core.fsmonitor=false", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def rescore(run: Path) -> dict[str, Any]:
    from core.subject.recording import load_recording
    from core.subject.synergy import kraskov_suite, synergy_suite

    report_path = run / "subject_core_report.json"
    report = json.loads(report_path.read_text())
    seed = int(report.get("campaign", {}).get("frozen", {}).get("seed", 0) or 0)
    turns = load_recording(run).by_turn()
    recorded = report.get("synergy_v4") or []
    again = [item.as_dict() for item in synergy_suite(turns, seed=seed, of="change", clocks_out=True)]
    for was, now in zip(recorded, again, strict=False):
        if abs(float(was.get("synergy", 0.0)) - float(now.get("synergy", 0.0))) > ROUNDING:
            raise SystemExit(
                f"{run}: the v3 reading recomputed from the recording does not match the report "
                f"({was.get('sources')} -> {was.get('target')}: {was.get('synergy')} against {now.get('synergy')})"
            )
    rows = [item.as_dict() for item in kraskov_suite(turns, seed=seed)]
    return {
        "run": str(run),
        "seed": seed,
        "commit": _commit(),
        "v3_recomputed_matches": bool(recorded),
        "kraskov": rows,
        "passes": sum(1 for row in rows if row["passes"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    for run in args.runs:
        result = rescore(run)
        (run / SIDECAR).write_text(json.dumps(result, indent=2) + "\n")
        for row in result["kraskov"]:
            print(
                f"{run.name} {','.join(row['sources'])}->{row['target']}: synergy {row['synergy']:+.4f} "
                f"shift bar {row['shift_bar']:+.4f} additive bar {row['additive_bar']:+.4f} passes {row['passes']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
