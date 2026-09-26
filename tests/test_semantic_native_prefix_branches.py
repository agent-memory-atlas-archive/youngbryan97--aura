"""Source reuse preserves evidence boundaries and the ordering of alternatives."""

from types import SimpleNamespace

import pytest

from core.learning.frozen_prefix_branches import native_source_anchor
from tools.probe_semantic_native_prefix_branches import (
    branch_source_groups,
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
