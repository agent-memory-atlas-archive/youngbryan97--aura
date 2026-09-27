"""Source reuse preserves evidence boundaries and the ordering of alternatives."""

from types import SimpleNamespace

import pytest

from core.learning.frozen_prefix_branches import native_source_anchor
from tools.probe_semantic_native_prefix_branches import (
    branch_choice_partitions,
    branch_source_groups,
    partitioned_score_equivalence,
    ranked_score_equivalence,
)


def test_source_anchor_excludes_every_continuation_even_with_different_boundaries():
    left = SimpleNamespace(tokens=(1, 2, 3, 4), continuation_start=2)
    right = SimpleNamespace(tokens=(1, 2, 3, 7), continuation_start=3)
    assert native_source_anchor((left, right)) == (1, 2)
    with pytest.raises(ValueError, match="differs"):
        native_source_anchor((left, SimpleNamespace(tokens=(1, 9, 3, 4), continuation_start=2)))
    with pytest.raises(ValueError, match="boundary"):
        native_source_anchor(())


def test_numerical_tolerance_does_not_authorize_changed_winners_or_lower_ranks():
    assert ranked_score_equivalence((-1., -2., -3.), (-1.01, -2., -3.), tolerance=.02)["accepted"] is True
    winner = ranked_score_equivalence((-1., -1.01), (-1.02, -1.01), tolerance=.1)
    assert winner["accepted"] is False and winner["winner_preserved"] is False
    lower = ranked_score_equivalence((-1., -2., -2.01), (-1., -2.02, -2.01), tolerance=.1)
    assert lower["winner_preserved"] is True and lower["accepted"] is False
    with pytest.raises(ValueError, match="finite matched"):
        ranked_score_equivalence((-1.,), (float("nan"),), tolerance=.1)


def test_source_probe_groups_only_matched_alternatives_and_never_filters_by_success():
    supervision = {"rows": [{"source": source, "program_sha256": program,
        "tokens": [1, 2, 3], "continuation_start": 2, "semantic_positions": [2]}
        for source, program in (("b", "p2"), ("b", "p1"), ("a", "p1"))]}
    groups = branch_source_groups(supervision, count=1)
    assert len(groups) == 1 and groups[0][0] == "b" and len(groups[0][1]) == 2
    with pytest.raises(ValueError, match="complete source-bound"):
        branch_source_groups(supervision, count=2)
    with pytest.raises(ValueError, match="finite source"):
        branch_source_groups(supervision, count=True)


def _grammar_supervision():
    return {"grammar_choice_contract": {"basis": "native_typed_grammar_teacher_choices_v1"},
        "rows": [{"source": "a", "decision_index": decision, "choice_index": choice,
                  "kind": kind, "correct_index": 0, "tokens": [1, 2, 3 + choice],
                  "continuation_start": 2, "semantic_positions": [2]}
                 for decision, kind, choices in ((0, "operation", 2), (1, "reference", 3), (2, "finish", 2))
                 for choice in range(choices)]}


def test_grammar_probe_keeps_every_choice_and_partitions_by_real_competition():
    supervision = _grammar_supervision()
    groups = branch_source_groups(supervision, count=1)
    assert len(groups) == 1 and len(groups[0][1]) == 7
    partitions = branch_choice_partitions(supervision, "a")
    assert [row["indices"] for row in partitions] == [(0, 1), (2, 3, 4), (5, 6)]
    assert [row["kind"] for row in partitions] == ["operation", "reference", "finish"]
    reference = (-1., -2., -1.01, -2.01, -3., -1.02, -2.02)
    changed = (-1.03, -2.03, -1., -2., -3., -1.02, -2.02)
    assert ranked_score_equivalence(reference, changed, tolerance=.1)["accepted"] is False
    assert all(row["comparison"]["accepted"] for row in partitioned_score_equivalence(
        reference, changed, partitions, tolerance=.1))


@pytest.mark.parametrize("removed", [0, 3, 4])
def test_grammar_probe_refuses_missing_choice_or_decision_groups(removed):
    supervision = _grammar_supervision()
    if removed == 4:
        supervision["rows"] = [row for row in supervision["rows"] if row["decision_index"] != 1]
    else:
        supervision["rows"].pop(removed)
    with pytest.raises(ValueError, match="missing|incomplete"):
        branch_choice_partitions(supervision, "a")
