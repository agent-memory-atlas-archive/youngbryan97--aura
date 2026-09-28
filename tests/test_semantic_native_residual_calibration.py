"""A residual cannot be selected by hiding a baseline path regression."""

from __future__ import annotations

import json

import pytest

from core.learning.semantic_native_path_calibration import native_path_profile
from tools.calibrate_semantic_native_residual import (
    measured_row,
    residual_admission,
    residual_scales,
)
from tools.evaluate_semantic_native_checkpoint import digest


def _row(source: str, correct: bool) -> dict:
    return {"source": source, "decisions": [
        {"kind": "operation", "correct_index": 0,
         "scores": [2., 1.] if correct else [1., 2.]},
        {"kind": "termination", "correct_index": 0, "scores": [0.]},
    ]}


def test_residual_requires_ordered_endpoints_and_an_interior_scale():
    assert residual_scales([0., 0.08, 1.]) == (0., 0.08, 1.)
    for bad in ([0., 1.], [0.08, 1.], [0., 0.08], [0., 0.08, 0.08, 1.],
                [0., 1., 0.08], [0., float("nan"), 1.], [0., 1.01, 1.]):
        with pytest.raises(ValueError, match="ordered 0"):
            residual_scales(bad)


def test_residual_selects_source_gain_only_when_all_baseline_paths_survive():
    rows = {0.: [_row("a", True), _row("b", False)],
            0.1: [_row("a", False), _row("b", True)],
            0.2: [_row("a", True), _row("b", True)]}
    result = residual_admission(rows, ["a", "b"])
    assert result["selected_scale"] == 0.2
    assert [row["eligible"] for row in result["adjudication"]] == [True, False, True]
    assert result["adjudication"][1]["lost_baseline_sources"] == ["a"]
    assert result["totals"]["0.2"]["exact_teacher_paths"] == 2


def test_residual_returns_baseline_when_every_fitted_scale_regresses():
    rows = {0.: [_row("a", True), _row("b", False)],
            0.1: [_row("a", False), _row("b", True)],
            1.: [_row("a", False), _row("b", True)]}
    assert residual_admission(rows, ["a", "b"])["selected_scale"] == 0.
    with pytest.raises(ValueError, match="source coverage"):
        residual_admission(rows, ["b", "a"])


def test_residual_row_is_bound_to_scale_and_teacher_choices(tmp_path):
    decisions = [{"kind": "operation", "correct_index": 0,
                  "choices": ["add", "sub"], "scores": [2., 1.]},
                 {"kind": "termination", "correct_index": 0,
                  "choices": ["finish"], "scores": [0.]}]
    body = {"schema": "aura.native_residual_calibration_row.v1", "plan_sha256": "plan",
            "source": "source", "scale": 0.08, "decisions": decisions,
            "profile": native_path_profile(decisions)}
    path = tmp_path / "row.json"
    path.write_text(json.dumps({**body, "receipt_sha256": digest(body)}))
    choices = [[{"kind": "operation", "correct_index": 0, "choice": "add"},
                {"kind": "operation", "correct_index": 0, "choice": "sub"}],
               [{"kind": "termination", "correct_index": 0, "choice": "finish"}]]
    assert measured_row(path, plan={"plan_sha256": "plan"}, source="source",
                        scale=0.08, choices=choices)["profile"]["exact_teacher_path"]
    with pytest.raises(ValueError, match="identity differs"):
        measured_row(path, plan={"plan_sha256": "plan"}, source="source",
                     scale=0.09, choices=choices)
    choices[0][1]["choice"] = "mul"
    with pytest.raises(ValueError, match="alternatives differ"):
        measured_row(path, plan={"plan_sha256": "plan"}, source="source",
                     scale=0.08, choices=choices)
