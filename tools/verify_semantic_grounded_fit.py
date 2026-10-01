#!/usr/bin/env python3
"""Independently check a grounded fit checkpoint; never score held language."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def verify(directory):
    from core.learning.semantic_grounded_binding_engine import verify_grounded_fit_checkpoint
    from core.runtime.file_read_gateway import read_stable_bytes

    directory = Path(directory)
    report = verify_grounded_fit_checkpoint(directory)
    acquisition = None
    if report.get("joint_native_adapter_training"):
        custody = directory.parent / (directory.name + "-native-custody")
        native = json.loads(read_stable_bytes(custody / "plan.json", max_bytes=16 * 1024 ** 2))
        plan = {key: value for key, value in native.items() if key != "plan_sha256"}
        if (plan != report["native_contract"] or hashlib.sha256(json.dumps(plan, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest() != native.get("plan_sha256")):
            raise ValueError("grounded native verification plan differs")
        acquisition = json.loads(read_stable_bytes(custody / "completion.json", max_bytes=16 * 1024 ** 2))
        if (acquisition.get("schema") != "aura.grounded_native_acquisition.v1"
                or acquisition.get("plan_sha256") != native["plan_sha256"]
                or acquisition.get("fit_receipt_sha256") != report["receipt_sha256"]
                or acquisition.get("held_sources_scored") is not False
                or acquisition.get("serving_authority") is not False):
            raise ValueError("grounded native verification completion differs")
    body = {"schema": "aura.grounded_binding_verification.v1", "fit_identity": report["fit_identity"],
        "fit_receipt_sha256": report["receipt_sha256"], "weights_sha256": report["weights_sha256"],
        "selected_step": report["selected_step"], "completed_updates": report["steps"],
        "source_fit_count": len(report["supervision"]["training"]),
        "source_calibration_count": len(report["supervision"]["calibration"]),
        "native_acquisition_sha256": hashlib.sha256(json.dumps(acquisition, sort_keys=True,
            separators=(",", ":")).encode()).hexdigest() if acquisition is not None else None,
        "artifacts_verified": True, "current_implementation_drift": [],
        "learned_checkpoint_selected": report["selected_step"] > 0,
        "model_weights_loaded": False, "held_sources_scored": False,
        "semantic_success": None, "qualification_evidence": False, "serving_authority": False}
    return {**body, "receipt_sha256": hashlib.sha256(json.dumps(body, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.directory)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["learned_checkpoint_selected"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
