"""Path calibration uses the decoder's tie rule and cannot hide binding failures."""

from copy import deepcopy

import pytest

from core.learning.procedure_induction import Instruction, Program
from core.learning.semantic_native_decision_supervision import native_teacher_decisions
from core.learning.semantic_native_grammar import (
    NativeGrammarIncompleteError,
    decode_native_grammar,
)
from core.learning.semantic_native_path_calibration import native_path_profile, native_path_totals
from tools.audit_semantic_native_path_calibration import (
    calibration_competitions,
    verified_measurement,
)


def path():
    return [{"kind": "operation", "scores": [-10., 0.], "correct_index": 1},
            {"kind": "reference", "scores": [0., -10.], "correct_index": 0},
            {"kind": "termination", "scores": [0., -10.], "correct_index": 0}]


def test_one_wrong_binding_makes_the_whole_path_inexact():
    decisions = path()
    decisions[1]["scores"] = [0., .01]
    result = native_path_profile(decisions)
    assert not result["exact_teacher_path"]
    assert result["first_error"]["kind"] == "reference"
    assert result["correct_counts"] == {"operation": 1, "reference": 0, "termination": 1}
    assert result["free_decode_measured"] is False


def test_ties_match_first_index_decoder_rule_and_stable_extreme_scores():
    decisions = path()
    decisions[0]["scores"] = [1e8, 1e8]
    assert native_path_profile(decisions)["first_error"]["chosen_index"] == 0
    decisions[0]["correct_index"] = 0
    assert native_path_profile(decisions)["exact_teacher_path"]


@pytest.mark.parametrize("defect", ["empty", "nan", "bool", "index", "kind", "finish"])
def test_invalid_or_incomplete_competitions_are_not_passing(defect):
    decisions = path()
    if defect == "empty":
        decisions = []
    elif defect == "nan":
        decisions[0]["scores"][0] = float("nan")
    elif defect == "bool":
        decisions[0]["scores"][0] = True
    elif defect == "index":
        decisions[0]["correct_index"] = True
    elif defect == "kind":
        decisions[0]["kind"] = "label"
    else:
        decisions.pop()
    with pytest.raises(ValueError):
        native_path_profile(decisions)


def test_totals_require_complete_unique_ordered_source_population():
    rows = [{"source": "a", "decisions": path()}, {"source": "b", "decisions": path()}]
    bad = deepcopy(rows)
    bad[1]["decisions"][1]["scores"] = [0., 1.]
    totals = native_path_totals(bad, ["a", "b"])
    assert totals["exact_teacher_paths"] == 1
    assert totals["first_error_counts"] == {"reference": 1}
    for sources in (["b", "a"], ["a", "a"], ["a"]):
        with pytest.raises(ValueError, match="coverage"):
            native_path_totals(rows, sources)


def supervision_fixture():
    training = {"plan_sha256": "training", "objective": "grammar_choices", "fit_ids": ["fit"],
                "captured_fit_ids": ["fit"], "calibration_ids": ["cal"], "held_ids": ["held"]}
    rows = [{"source": source, "decision_index": ordinal, "kind": kind,
             "correct_index": 0, "choice_index": index, "choice": choice}
            for source in ("fit", "cal")
            for ordinal, kind, choices in ((0, "operation", ("sub", "add")),
                                          (1, "termination", ("finish", "continue")))
            for index, choice in enumerate(choices)]
    return training, {"plan_sha256": "training", "rows": rows}


@pytest.mark.parametrize("defect", ["held", "missing_source", "missing_choice", "duplicate_choice", "finish", "label"])
def test_calibration_inventory_cannot_admit_held_or_incomplete_teacher_data(defect):
    training, supervision = supervision_fixture()
    assert set(calibration_competitions(training, supervision)) == {"cal"}
    if defect == "held":
        training["held_ids"] = ["cal"]
    elif defect == "missing_source":
        supervision["rows"] = [row for row in supervision["rows"] if row["source"] != "cal"]
    elif defect == "missing_choice":
        supervision["rows"].pop(-2)
    elif defect == "duplicate_choice":
        supervision["rows"].append(deepcopy(supervision["rows"][-1]))
    elif defect == "finish":
        supervision["rows"][-2]["choice"] = "continue"
    else:
        supervision["rows"][-1]["correct_index"] = 1
    with pytest.raises(ValueError):
        calibration_competitions(training, supervision)


def test_verification_recomputes_profile_and_preserves_measured_alternative_identity():
    training, supervision = supervision_fixture()
    choices = calibration_competitions(training, supervision)["cal"]
    decisions = [{"kind": group[0]["kind"], "correct_index": 0,
                  "choices": [row["choice"] for row in group], "scores": [0., -1.]}
                 for group in choices]
    measured = {"plan_sha256": "audit", "source": "cal", "checkpoint_receipt_sha256": "checkpoint",
                "decisions": decisions, "profile": native_path_profile(decisions)}
    kwargs = {"plan": {"plan_sha256": "audit"}, "checkpoint": {"receipt_sha256": "checkpoint"},
              "source": "cal", "competitions": choices}
    verified_measurement(measured, **kwargs)
    measured["decisions"][0]["scores"] = [-2., 0.]
    with pytest.raises(ValueError, match="profile"):
        verified_measurement(measured, **kwargs)
    measured["profile"] = native_path_profile(measured["decisions"])
    measured["decisions"][0]["choices"].reverse()
    with pytest.raises(ValueError, match="alternatives"):
        verified_measurement(measured, **kwargs)


@pytest.mark.parametrize("bad_ordinal", [None, 0, 1, 2, 3, 4, 5, 6])
def test_complete_teacher_competition_wins_iff_same_scorer_greedily_reconstructs_target(bad_ordinal):
    target = Program(2, (Instruction("sub", (0, 1)), Instruction("mul", (2, 1))))
    teachers = native_teacher_decisions(target, ("integer", "integer"))
    measured = []
    by_prefix = {}
    for ordinal, teacher in enumerate(teachers):
        scores = [-2.] * len(teacher.choices)
        scores[teacher.correct_index] = 0.
        if ordinal == bad_ordinal and len(scores) > 1:
            scores[(teacher.correct_index + 1) % len(scores)] = 1.
        measured.append({"kind": teacher.kind, "correct_index": teacher.correct_index,
                         "scores": scores})
        by_prefix[tuple(choice.text for choice in teacher.choices)] = tuple(scores)
    try:
        generated = decode_native_grammar(("integer", "integer"),
            lambda choices: by_prefix.get(tuple(choice.text for choice in choices),
                                          tuple(-float(index) for index in range(len(choices)))),
            max_steps=3)
        exact = generated.program == target
    except NativeGrammarIncompleteError:
        exact = False
    assert native_path_profile(measured)["exact_teacher_path"] == exact
