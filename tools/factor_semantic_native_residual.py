#!/usr/bin/env python3
"""Select decision-kind residuals from already measured source-calibration paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def factor_residual(calibration_directory, training_directory):
    from core.learning.semantic_native_factorized_residual import (
        FACTORIZED_RESIDUAL_CONTRACT,
        factorized_residual_admission,
    )
    from tools.evaluate_semantic_native_checkpoint import verified_document
    from tools.verify_semantic_native_residual import verify_residual

    verified = verify_residual(calibration_directory, training_directory)
    if verified["current_implementation_drift"]:
        raise ValueError("native factorization source measurement has implementation drift")
    plan = verified_document(calibration_directory / "plan.json", "plan_sha256")
    rows = {scale: [verified_document(calibration_directory / "rows" / f"{scale:g}-{source}.json")
                    for source in plan["sources"]] for scale in plan["scales"]}
    admission = factorized_residual_admission(rows, plan["sources"])
    paths = ("core/learning/semantic_native_factorized_residual.py",
             "tools/factor_semantic_native_residual.py")
    return {"schema": "aura.native_factorized_residual.v1",
            "contract": dict(FACTORIZED_RESIDUAL_CONTRACT),
            "calibration_directory": str(calibration_directory.resolve()),
            "calibration_plan_sha256": plan["plan_sha256"],
            "calibration_report_receipt_sha256": verified["report_receipt_sha256"],
            "training_plan_sha256": plan["training_plan_sha256"],
            "candidate_checkpoint_receipt_sha256": plan["candidate_checkpoint_receipt_sha256"],
            "baseline_checkpoint_receipt_sha256": plan["baseline_checkpoint_receipt_sha256"],
            "sources": plan["sources"], **admission,
            "implementation": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                               for name in paths},
            "held_labels_used": False, "serving_authority": False,
            "qualification_evidence": False, "free_decode_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-directory", type=Path, required=True)
    parser.add_argument("--training-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.probe_semantic_proposer_crossfit import _save_if_absent
    from tools.refit_semantic_argument_proposals import configure_refit_environment

    configure_refit_environment(args.output)
    body = factor_residual(args.calibration_directory, args.training_directory)
    _save_if_absent(args.output, {**body, "receipt_sha256": digest(body)})
    print(json.dumps({"selected_scales": body["selected_scales"],
                      "selected_totals": body["selected_totals"], "receipt_sha256": digest(body)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
