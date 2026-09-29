"""A residual cannot be selected by hiding a baseline path regression."""

from __future__ import annotations

import json

import pytest

from core.learning.semantic_native_path_calibration import native_path_profile
from tools.calibrate_semantic_native_residual import (
    graph_competitions,
    measured_row,
    residual_admission,
    residual_scales,
    residual_training_contract,
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


def test_typed_contrast_fit_uses_existing_rejected_path_residual_gate():
    from core.learning.semantic_native_path_objective import GRAMMAR_PATH_CONTRACT
    from core.learning.semantic_native_typed_source_pairs import TYPED_SOURCE_PAIR_CONTRACT

    plan = {"schema": "aura.semantic_native_fit_plan.v6", "suffix_layers": 1,
            "reused_prefix_contract": {},
            "grammar_source_pair_contract": dict(TYPED_SOURCE_PAIR_CONTRACT),
            "grammar_path_objective_contract": dict(GRAMMAR_PATH_CONTRACT),
            "contrast_policy": "all_native_type_admitted_teacher_decisions_v1"}
    residual_training_contract({"schema": "aura.semantic_native_fit_plan.v5",
                                "suffix_layers": 1, "reused_prefix_contract": {}}, {"step": 0})
    residual_training_contract(plan, {"step": 0})
    for changed_plan, selected in (
        ({**plan, "schema": "aura.semantic_native_fit_plan.v7"}, {"step": 0}),
        ({**plan, "grammar_source_pair_contract": {}}, {"step": 0}),
        ({**plan, "grammar_path_objective_contract": {}}, {"step": 0}),
        ({**plan, "contrast_policy": "operation_only"}, {"step": 0}),
        ({**plan, "suffix_layers": 2}, {"step": 0}),
        (plan, {"step": 101}),
    ):
        with pytest.raises(ValueError, match="native residual"):
            residual_training_contract(changed_plan, selected)


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


def test_joint_graph_residual_keeps_both_baseline_success_sets():
    from core.learning.semantic_native_path_objective import (
        GRAMMAR_PATH_CONTRACT, JOINT_GRAPH_CONTRAST_CONTRACT,
    )
    from core.learning.semantic_native_typed_source_pairs import TYPED_SOURCE_PAIR_CONTRACT

    plan = {"schema": "aura.semantic_native_fit_plan.v7", "suffix_layers": 1,
            "reused_prefix_contract": {}, "joint_graph_contrast_limit": 3,
            "grammar_source_pair_contract": dict(TYPED_SOURCE_PAIR_CONTRACT),
            "grammar_path_objective_contract": dict(GRAMMAR_PATH_CONTRACT),
            "graph_contrast_contract": dict(JOINT_GRAPH_CONTRAST_CONTRACT),
            "contrast_policy": "all_native_type_admitted_teacher_decisions_v1"}
    residual_training_contract(plan, {"step": 0})
    supervision = {"graph_contrast_contract": dict(JOINT_GRAPH_CONTRAST_CONTRACT), "graph_rows": [
        {"source": source, "choice_index": index, "positive": index == 0,
         "program_sha256": f"{source}-{index}"}
        for source in ("a", "b") for index in range(2)]}
    assert len(graph_competitions(plan, supervision, ["a", "b"])) == 2
    supervision["graph_rows"].pop()
    with pytest.raises(ValueError, match="graph competition differs"):
        graph_competitions(plan, supervision, ["a", "b"])

    def row(source, path_correct, graph_correct):
        return {**_row(source, path_correct), "whole_graph": {
            "program_sha256s": [source + "-0", source + "-1"], "positive_index": 0,
            "scores": [2., 1.] if graph_correct else [1., 2.]}}

    rows = {0.: [row("a", True, True), row("b", False, False)],
            0.1: [row("a", True, False), row("b", True, True)],
            0.2: [row("a", True, True), row("b", True, True)]}
    result = residual_admission(rows, ["a", "b"])
    assert result["selected_scale"] == 0.2
    assert result["adjudication"][1]["lost_baseline_graph_sources"] == ["a"]
    assert result["adjudication"][1]["eligible"] is False
    rows[0.2][0].pop("whole_graph")
    with pytest.raises(ValueError, match="graph measurements are incomplete"):
        residual_admission(rows, ["a", "b"])
