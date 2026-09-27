#!/usr/bin/env python3
"""Freeze a development-selected CAA cell without touching sealed evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_vector_identity(result: dict) -> None:
    identity = result.get("progress_identity")
    if not isinstance(identity, dict):
        raise ValueError("caa_development_progress_identity_missing")
    root = Path(result["vectors"])
    if not root.is_dir() or not isinstance(identity.get("vectors"), dict):
        raise ValueError("caa_development_vectors_missing")
    for name, expected in identity["vectors"].items():
        if _sha(root / name) != expected:
            raise ValueError("caa_development_vectors_drifted")
    if _sha(root / "metadata.json") != identity.get("metadata_sha256"):
        raise ValueError("caa_development_metadata_drifted")
    null_root = Path(result.get("polarity_null_vectors", ""))
    if not null_root.is_dir() or not identity.get("polarity_null_vectors"):
        raise ValueError("caa_development_null_vectors_missing")
    for name, expected in identity["polarity_null_vectors"].items():
        if _sha(null_root / name) != expected:
            raise ValueError("caa_development_null_vectors_drifted")
    if _sha(null_root / "metadata.json") != identity.get("polarity_null_metadata_sha256"):
        raise ValueError("caa_development_null_metadata_drifted")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-samples", type=int, default=24)
    parser.add_argument("--max-accuracy-loss", type=float, default=0.0)
    parser.add_argument("--max-margin-loss", type=float, default=0.0)
    parser.add_argument("--max-null-score-shift", type=float, default=0.25)
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("development selection output already exists")

    from core.consciousness.caa.development_selection import (
        DevelopmentBar,
        evaluate_development_result,
        select_development_candidate,
    )
    from tools.run_caa_steering_campaign import DEVELOPMENT_TASKS

    bar = DevelopmentBar(args.min_samples, args.max_accuracy_loss,
                         args.max_margin_loss, args.max_null_score_shift)
    verdicts = []
    for path in args.result:
        raw = json.loads(path.read_text(encoding="utf-8"))
        _check_vector_identity(raw)
        verdict = evaluate_development_result(raw, expected_tasks=DEVELOPMENT_TASKS, bar=bar)
        verdicts.append({**verdict, "result_sha256": _sha(path), "result_path": str(path)})
    decision = select_development_candidate(verdicts)
    decision["bar"] = {"min_samples": bar.min_samples,
                       "max_accuracy_loss": bar.max_accuracy_loss,
                       "max_margin_loss": bar.max_margin_loss,
                       "max_null_score_shift": bar.max_null_score_shift}
    decision["verdicts"] = verdicts
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": decision["selected"] is not None,
                      "eligible_cells": decision["eligible_cells"],
                      "out": str(args.out)}, sort_keys=True), flush=True)
    return 0 if decision["selected"] is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
