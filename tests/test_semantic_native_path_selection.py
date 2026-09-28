"""A lower average loss cannot justify discarding a measured baseline path."""

from copy import deepcopy

import pytest

from core.learning.semantic_native_path_selection import (
    select_native_joint_graph_checkpoint,
    select_native_path_checkpoint,
)


def measured(successes):
    return [{"source": source, "decisions": [
        {"kind": "operation", "correct_index": 0, "scores": [0., -1.] if success else [-1., 0.]},
        {"kind": "termination", "correct_index": 0, "scores": [0., -1.]}]}
        for source, success in zip(("a", "b", "c"), successes, strict=True)]


def test_less_loss_and_more_total_successes_cannot_buy_one_baseline_regression():
    checkpoints = [{"step": 0, "calibration_loss": 1.}, {"step": 1, "calibration_loss": .01}]
    measurements = {0: measured((True, False, False)), 1: measured((False, True, True))}
    selected, verdicts = select_native_path_checkpoint(checkpoints, measurements, ["a", "b", "c"])
    assert selected["step"] == 0
    assert verdicts[1]["baseline_path_regressions"] == ["a"]
    assert not verdicts[1]["eligible"]


def test_true_superset_gain_is_selected_before_average_loss():
    checkpoints = [{"step": 0, "calibration_loss": .01}, {"step": 1, "calibration_loss": .2},
                   {"step": 2, "calibration_loss": .3}]
    measurements = {0: measured((True, False, False)), 1: measured((True, True, False)),
                    2: measured((True, True, True))}
    selected, _ = select_native_path_checkpoint(checkpoints, measurements, ["a", "b", "c"])
    assert selected["step"] == 2


def test_tie_uses_loss_then_earliest_checkpoint_without_held_information():
    checkpoints = [{"step": 0, "calibration_loss": .3}, {"step": 1, "calibration_loss": .2},
                   {"step": 2, "calibration_loss": .2}]
    measurements = {step: measured((True, False, False)) for step in (0, 1, 2)}
    assert select_native_path_checkpoint(checkpoints, measurements, ["a", "b", "c"])[0]["step"] == 1


@pytest.mark.parametrize("defect", ["missing_baseline", "missing_path", "duplicate", "nan"])
def test_missing_or_invalid_measurement_is_not_eligible(defect):
    checkpoints = [{"step": 0, "calibration_loss": .3}, {"step": 1, "calibration_loss": .2}]
    measurements = {0: measured((True, False, False)), 1: measured((True, True, False))}
    if defect == "missing_baseline":
        measurements.pop(0)
    elif defect == "missing_path":
        measurements[1].pop()
    elif defect == "duplicate":
        checkpoints.append(deepcopy(checkpoints[1]))
    else:
        checkpoints[1]["calibration_loss"] = float("nan")
    with pytest.raises(ValueError):
        select_native_path_checkpoint(checkpoints, measurements, ["a", "b", "c"])


def joint_measured(path_successes, graph_successes):
    rows = measured(path_successes)
    for row, success in zip(rows, graph_successes, strict=True):
        row["whole_graph"] = {"positive_index": 0,
                              "scores": [1., 0.] if success else [0., 1.]}
    return rows


def test_joint_selection_rejects_graph_regression_even_if_paths_improve():
    checkpoints = [{"step": 0, "calibration_loss": 1.}, {"step": 1, "calibration_loss": .01}]
    measurements = {0: joint_measured((True, False, False), (True, False, False)),
                    1: joint_measured((True, True, True), (False, True, True))}
    selected, verdicts = select_native_joint_graph_checkpoint(
        checkpoints, measurements, ["a", "b", "c"])
    assert selected["step"] == 0
    assert verdicts[1]["baseline_graph_regressions"] == ["a"]
    assert not verdicts[1]["eligible"]


def test_joint_selection_prefers_graph_gain_over_lower_loss_without_regression():
    checkpoints = [{"step": 0, "calibration_loss": .1}, {"step": 1, "calibration_loss": .5}]
    measurements = {0: joint_measured((True, False, False), (True, False, False)),
                    1: joint_measured((True, False, False), (True, True, False))}
    selected, verdicts = select_native_joint_graph_checkpoint(
        checkpoints, measurements, ["a", "b", "c"])
    assert selected["step"] == 1
    assert verdicts[1]["positive_graph_rankings"] == 2


def test_joint_selection_does_not_count_a_tied_graph_as_a_win():
    checkpoints = [{"step": 0, "calibration_loss": 1.}, {"step": 1, "calibration_loss": .01}]
    measurements = {0: joint_measured((False, False, False), (True, False, False)),
                    1: joint_measured((False, False, False), (True, False, False))}
    measurements[1][0]["whole_graph"]["scores"] = [0., 0.]
    selected, verdicts = select_native_joint_graph_checkpoint(
        checkpoints, measurements, ["a", "b", "c"])
    assert selected["step"] == 0
    assert verdicts[1]["baseline_graph_regressions"] == ["a"]


@pytest.mark.parametrize("defect", ["missing", "nan", "wrong_positive", "short"])
def test_joint_selection_requires_every_graph_measurement(defect):
    checkpoints = [{"step": 0, "calibration_loss": .1}]
    rows = joint_measured((True, False, False), (True, False, False))
    graph = rows[0]["whole_graph"]
    if defect == "missing":
        del rows[0]["whole_graph"]
    elif defect == "nan":
        graph["scores"][0] = float("nan")
    elif defect == "wrong_positive":
        graph["positive_index"] = 1
    else:
        graph["scores"].pop()
    with pytest.raises(ValueError, match="whole-graph"):
        select_native_joint_graph_checkpoint(checkpoints, {0: rows}, ["a", "b", "c"])
