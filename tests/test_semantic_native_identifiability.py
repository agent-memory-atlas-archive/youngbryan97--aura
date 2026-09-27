"""A scoring-input collision can make an exact teacher task impossible."""

from copy import deepcopy

import pytest

from tools.verify_semantic_native_fit import audit_grammar_identifiability


def decision(source="a", correct=0):
    return [{"source": source, "decision_index": 0, "choice_index": index,
             "correct_index": correct, "kind": "reference", "tokens": [1, 2, 3 + index, 99],
             "continuation_start": 2, "semantic_positions": [2]}
            for index in range(2)]


def test_identical_causal_inputs_cannot_require_opposite_labels():
    rows = decision() + decision("b", 1)
    # Target metadata and future unscored tokens cannot resolve this collision.
    for row in rows[2:]:
        row["target_program_sha256"] = "different"
        row["tokens"][-1] = 100
    result = audit_grammar_identifiability(rows)
    assert result["decision_groups"] == 2
    assert result["unique_scoring_inputs"] == 1
    assert result["contradictory_scoring_input_groups"] == 1
    assert result["maximum_exact_teacher_decisions"] == 1
    assert result["learnability_proven"] is result["semantic_transfer_proven"] is False


def test_identical_inputs_with_same_label_are_not_contradictory():
    result = audit_grammar_identifiability(decision() + decision("b"))
    assert result["duplicate_scoring_input_groups"] == 1
    assert result["contradictory_scoring_input_groups"] == 0
    assert result["maximum_exact_teacher_decisions"] == 2


def test_distinguishing_context_and_scoring_positions_remain_visible():
    left, right = decision(), decision("b", 1)
    for row in right:
        row["tokens"][0] = 42
    result = audit_grammar_identifiability(left + right)
    assert result["unique_scoring_inputs"] == 2
    for row in right:
        row["tokens"][0] = 1
        row["semantic_positions"] = [2, 3]
    assert audit_grammar_identifiability(left + right)["unique_scoring_inputs"] == 2


@pytest.mark.parametrize("defect", ["empty", "duplicate", "missing", "label", "kind", "position",
                                    "future", "tokens", "boundary", "identity", "boolean"])
def test_malformed_groups_are_not_evidence(defect):
    rows = deepcopy(decision())
    if defect == "empty":
        rows = []
    elif defect == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif defect == "missing":
        rows.pop(0)
    elif defect == "label":
        rows[0]["correct_index"] = 1
    elif defect == "kind":
        rows[0]["kind"] = "other"
    elif defect == "position":
        rows[0]["semantic_positions"] = [2, 2]
    elif defect == "future":
        rows[0]["semantic_positions"] = [4]
    elif defect == "tokens":
        rows[0]["tokens"][0] = -1
    elif defect == "boundary":
        rows[0]["continuation_start"] = 0
    elif defect == "identity":
        rows[0]["decision_index"] = -1
    else:
        rows[0]["correct_index"] = False
    with pytest.raises(ValueError):
        audit_grammar_identifiability(rows)
