"""Choose development tissue without losing any baseline-correct calibration path."""

from __future__ import annotations

import math

from core.learning.semantic_native_path_calibration import native_path_profile, native_path_totals

PATH_SELECTION_CONTRACT = {
    "schema": "aura.native_path_checkpoint_selection.v1",
    "partition": "source_calibration_only",
    "baseline": "unfitted_step_zero",
    "admission": "no_lost_baseline_exact_teacher_paths",
    "ordering": "most_exact_paths_then_lowest_calibration_loss_then_earliest_step",
    "unseen_nonregression_guaranteed": False,
    "held_labels_used": False,
}


def select_native_path_checkpoint(checkpoints: list[dict], measurements: dict[int, list[dict]],
                                  sources: list[str]) -> tuple[dict, list[dict]]:
    """Keep unknown/unmeasured paths out of selection, including baseline zero."""
    if (not checkpoints or {row["step"] for row in checkpoints} != set(measurements)
            or len(checkpoints) != len(measurements) or 0 not in measurements
            or any(type(row["step"]) is not int or row["step"] < 0
                   or type(row["calibration_loss"]) not in {int, float}
                   or not math.isfinite(row["calibration_loss"]) for row in checkpoints)):
        raise ValueError("native path selection needs complete measured checkpoints and baseline zero")
    totals = {step: native_path_totals(rows, sources) for step, rows in measurements.items()}
    successes = {step: {row["source"] for row in rows
                       if native_path_profile(row["decisions"])["exact_teacher_path"]}
                 for step, rows in measurements.items()}
    baseline = successes[0]
    adjudication = [{"step": row["step"], "exact_teacher_paths": totals[row["step"]]["exact_teacher_paths"],
                     "baseline_path_regressions": sorted(baseline - successes[row["step"]]),
                     "eligible": baseline <= successes[row["step"]]}
                    for row in sorted(checkpoints, key=lambda row: row["step"])]
    eligible = {row["step"] for row in adjudication if row["eligible"]}
    selected = min((row for row in checkpoints if row["step"] in eligible),
                   key=lambda row: (-totals[row["step"]]["exact_teacher_paths"],
                                    row["calibration_loss"], row["step"]))
    return selected, adjudication
