"""Partial analogy preserves one-to-one bindings and never invents matches."""

import pytest

from core.cognition.relational_generalization import RelationalCase, RelationalGeneralizer
from core.cognition.structure_mapping import (
    Graph,
    Relation,
    _score,
    map_structures,
    map_structures_alternatives,
)


def test_larger_source_retains_a_real_partial_analogy():
    source = Graph("source", (Relation("follows", ("b", "a")), Relation("follows", ("c", "b"))))
    target = Graph("target", (Relation("behind", ("y", "x")),))
    match = map_structures(source, target)
    assert match is not None
    assert match.score == pytest.approx(0.5)
    assert len(set(match.mapping.values())) == len(match.mapping) == 2


def test_unmapped_object_cannot_match_by_surface_name():
    source = Graph("source", (Relation("r", ("a", "same")),))
    target = Graph("target", (Relation("r", ("x", "same")),))
    assert _score(source, target, {"a": "x"}, {"r": "r"})[0] == 0


def test_omitted_predicate_cannot_match_by_surface_name():
    source = Graph("source", (Relation("r", ("a",)),))
    target = Graph("target", (Relation("r", ("x",)),))
    assert _score(source, target, {"a": "x"}, {})[0] == 0
    assert _score(source, target, {"a": "x"}, None)[0] == 1


def test_relational_adapter_reuses_partial_mapping_without_claiming_equivalence():
    source = RelationalCase("queue", (("a", "person"), ("b", "person"), ("c", "person")),
                            (("follows", "b", "a"), ("follows", "c", "b")), goal="serve first")
    target = RelationalCase("road", (("x", "vehicle"), ("y", "vehicle")),
                            (("behind", "y", "x"),), goal="move first")
    engine = RelationalGeneralizer()
    assert engine.shared_structure(source, target).score == pytest.approx(0.5)
    assert not engine.same_problem(source, target)


def test_tied_role_mappings_are_retained_and_bounded():
    source = Graph("source", (Relation("near", ("center", "left")),
                              Relation("near", ("center", "right"))))
    target = Graph("target", (Relation("beside", ("hub", "east")),
                              Relation("beside", ("hub", "west"))))
    result = map_structures_alternatives(source, target)
    assert result.truncated is False
    assert {row.mapping["left"] for row in result.readings} == {"east", "west"}
    assert result.readings[0] == map_structures(source, target)
    bounded = map_structures_alternatives(source, target, max_results=1)
    assert len(bounded.readings) == 1
    assert bounded.truncated is True
    with pytest.raises(ValueError, match="budget"):
        map_structures_alternatives(source, target, max_results=0)


def test_tied_search_refuses_incomplete_predicate_enumeration():
    source = Graph("source", tuple(Relation(f"source_{index}", ("x",))
                                   for index in range(4)))
    target = Graph("target", tuple(Relation(f"target_{index}", ("y",))
                                   for index in range(10)))
    with pytest.raises(ValueError, match="exhaustive budget"):
        map_structures_alternatives(source, target)
    assert map_structures(source, target) is not None
