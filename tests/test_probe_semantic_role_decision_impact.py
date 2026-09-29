"""Saved-score role probes must count wins and losses without free-decode claims."""

import hashlib
from types import SimpleNamespace

from core.learning.semantic_role_rule_induction import RoleObservation, induce_role_rules
from tools.probe_semantic_role_decision_impact import calibration_impact


def test_saved_teacher_scores_count_a_real_first_reference_flip():
    source = "Subtract 4 from 32."
    other = "Remove 5 from 40."
    bank = induce_role_rules((
        RoleObservation(source, (0, 8), ((16, 18), (9, 10)), "a", "sub"),
        RoleObservation(other, (0, 6), ((14, 16), (7, 8)), "b", "sub"),
    ))
    identity = hashlib.sha256(source.encode()).hexdigest()
    decisions = [
        {"kind": "operation", "choices": ["sub"], "correct_index": 0, "scores": [0.]},
        {"kind": "reference", "choices": ["input:0", "input:1"],
         "correct_index": 1, "scores": [0., -1.]},
        {"kind": "reference", "choices": ["input:0", "input:1"],
         "correct_index": 0, "scores": [0., -1.]},
        {"kind": "termination", "choices": ["finish"], "correct_index": 0,
         "scores": [0.]},
    ]
    result = calibration_impact(bank, {identity: SimpleNamespace(source_text=source)},
                                {identity: {"decisions": decisions}}, strength=2.)
    assert result["population"] == 1
    assert result["affected_reference_competitions"] == 2
    assert result["new_exact_teacher_paths"] == 1
    assert result["lost_exact_teacher_paths"] == 0
    assert result["rows"][0]["baseline_first_error"]["kind"] == "reference"
    assert result["rows"][0]["candidate_first_error"] is None
    assert decisions[1]["scores"] == [0., -1.]
