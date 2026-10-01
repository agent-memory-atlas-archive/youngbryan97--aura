"""Context identities and joint evidence survive changes of presentation."""

from dataclasses import replace
from itertools import permutations, product
from types import SimpleNamespace

import numpy as np
import pytest

from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_argument_optimization import ArgumentOptimizationIncompleteError
from core.learning.semantic_context_binding import (
    BindingContext,
    BindingRelation,
    BindingRole,
    ContextReferent,
    bind_context_roles,
    binding_margin_survives,
    candidate_cost,
    source_identity_is_not_denotation,
)
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import RegisterUseContract


def context():
    return BindingContext((
        ContextReferent("env", "alice", "Person", "person:1", aliases=("Alice",),
                        attributes={"number": "singular"}),
        ContextReferent("env", "bob", "Person", "person:2", aliases=("Bob",)),
        ContextReferent("turn1", "result", "Number", "node:17", aliases=("previous result",)),
        ContextReferent("turn2", "literal", "Number", "token:9", aliases=("12",)),
    ), {"Entity": None, "Person": "Entity", "Number": "Entity"})


def test_namespace_and_aliases_do_not_collapse_equal_denotations():
    ctx = context()
    role = BindingRole("lhs", "minuend", "Number", referents=(("turn1", "result"),))
    result = bind_context_roles(ctx, [role])
    assert result.executable and result.bindings == (("lhs", ("turn1", "result")),)
    assert source_identity_is_not_denotation() == ()
    assert ctx.alias_candidates("PREVIOUS   result", role) == (("turn1", "result"),)
    assert ctx.alias_candidates("result", role) == ()


def test_known_absent_referent_never_binds_a_type_compatible_distractor():
    role = BindingRole("lhs", "value", "Number", referents=(), unbound_cost=1.)
    result = bind_context_roles(context(), [role])
    assert result.status == "partial" and result.bindings == (("lhs", None),)
    assert not result.executable


def test_no_evidence_is_unsupported_instead_of_a_random_embedding_match():
    result = bind_context_roles(context(), [BindingRole("lhs", "value", "Number")])
    assert result.status == "unsupported" and result.energy is None


def test_tied_same_type_candidates_remain_ambiguous():
    role = BindingRole("it", "theme", "Number")
    costs = {(role.identity, item.key): 0. for item in context().referents}
    result = bind_context_roles(context(), [role], costs=costs)
    assert result.status == "ambiguous" and result.margin == pytest.approx(0.)
    assert not result.executable and result.bindings != result.alternatives


def test_identity_can_fill_multiple_roles_unless_the_frame_declares_distinctness():
    key = ("turn1", "result")
    roles = [BindingRole("left", "minuend", "Number", referents=(key,)),
             BindingRole("right", "subtrahend", "Number", referents=(key,))]
    assert bind_context_roles(context(), roles).executable
    relation = BindingRelation("left", "right", "different")
    assert bind_context_roles(context(), roles, relations=[relation]).status == "infeasible"


def test_joint_binding_can_reject_both_local_winners():
    roles = [BindingRole("giver", "agent", "Person"), BindingRole("receiver", "goal", "Person")]
    costs = {("giver", ("env", "alice")): 0., ("giver", ("env", "bob")): 2.,
             ("receiver", ("env", "alice")): 0., ("receiver", ("env", "bob")): 1.}
    result = bind_context_roles(context(), roles, costs=costs,
                                relations=[BindingRelation("giver", "receiver", "different")])
    assert result.bindings == (("giver", ("env", "alice")), ("receiver", ("env", "bob")))
    assert result.margin == pytest.approx(1.)


def test_whole_assignment_filter_uses_exact_solver_cuts_and_distinct_admitted_runner_up():
    role = BindingRole("value", "theme", "Number")
    keys = (("turn1", "result"), ("turn2", "literal"))
    costs = {("value", keys[0]): 0., ("value", keys[1]): 1.}
    calls = []
    def admits(bindings):
        calls.append(bindings)
        return dict(bindings)["value"] == keys[1]
    result = bind_context_roles(context(), (role,), costs=costs, assignment_filter=admits)
    assert result.executable and result.bindings == (("value", keys[1]),)
    assert result.energy == 1. and result.margin is None and len(calls) == 2
    assert bind_context_roles(context(), (role,), costs=costs,
                              assignment_filter=lambda _: False).status == "infeasible"
    with pytest.raises(ArgumentOptimizationIncompleteError, match="filter_budget"):
        bind_context_roles(context(), (role,), costs=costs, assignment_filter=admits, assignment_limit=1)
    with pytest.raises(ValueError, match="explicit bool"):
        bind_context_roles(context(), (role,), costs=costs, assignment_filter=lambda _: 1)


def test_context_order_is_not_a_semantic_feature():
    roles = [BindingRole("giver", "agent", "Person", referents=(("env", "alice"),)),
             BindingRole("receiver", "goal", "Person", referents=(("env", "bob"),))]
    signatures = {bind_context_roles(replace(context(), referents=order), roles).bindings
                  for order in permutations(context().referents)}
    assert len(signatures) == 1


def test_scopes_availability_and_agreement_filter_before_scoring():
    value = ContextReferent("file", "report", "File", "file:1", scope=("chat1",),
                            available_from=2, available_until=4, attributes={"number": "plural"})
    ctx = BindingContext((value,), {"File": None})
    role = BindingRole("file", "target", "File", scope=("chat1", "turn3"), tick=3,
                       agreement={"number": "plural"}, referents=(value.key,))
    assert ctx.eligible(role, value)
    for bad in (replace(role, scope=("chat2",)), replace(role, tick=1), replace(role, tick=4),
                replace(role, agreement={"number": "singular"})):
        assert not ctx.eligible(bad, value)


def test_duplicate_display_aliases_retain_every_scoped_source():
    values = tuple(ContextReferent(namespace, "report", "File", namespace + ":report",
                                   scope=(namespace,), aliases=("report.pdf",)) for namespace in ("a", "b"))
    ctx = BindingContext(values, {"File": None})
    role = BindingRole("target", "file", "File", scope=("a", "turn1"))
    assert ctx.alias_candidates("report.pdf", role) == (("a", "report"),)
    with pytest.raises(ValueError, match="context"):
        BindingContext((values[0], values[0]), {"File": None})


def test_low_parser_confidence_uses_supplied_retrieval_without_lifting_type_constraints():
    role = BindingRole("it", "value", "Number")
    fallback = {("it", ("env", "alice")): -100., ("it", ("turn1", "result")): 1.,
                ("it", ("turn2", "literal")): 2.}
    result = bind_context_roles(context(), [role], fallback_costs=fallback,
                                parser_confidence=.2, parser_threshold=.7)
    assert result.executable and result.bindings == (("it", ("turn1", "result")),)
    assert bind_context_roles(context(), [role], parser_confidence=.2,
                              parser_threshold=.7).status == "unsupported"


def test_unknown_type_is_not_admitted_and_type_cycles_are_invalid():
    role = BindingRole("it", "theme", "Liquid")
    assert bind_context_roles(context(), [role]).status == "unsupported"
    with pytest.raises(ValueError, match="cycle"):
        BindingContext((), {"A": "B", "B": "A"})


def test_cosine_uses_real_vectors_without_overflow_or_global_rng_mutation():
    candidate = ContextReferent("env", "number", "Number", "token:0", embedding=(1e300, 1e300))
    ctx = BindingContext((candidate,), {"Number": None})
    role = BindingRole("it", "value", "Number", embedding=(1., 1.))
    before = np.random.get_state()
    assert candidate_cost(ctx, role, candidate) == pytest.approx(0., abs=1e-12)
    after = np.random.get_state()
    assert all(np.array_equal(left, right) for left, right in zip(before, after, strict=True))
    with pytest.raises(ValueError, match="nonzero"):
        replace(role, embedding=(0., 0.))


def test_declared_dependency_is_directional_and_not_assumed_from_role_names():
    ctx = replace(context(), edges=frozenset({(("turn1", "result"), ("turn2", "literal"))}))
    roles = [BindingRole("source", "source", "Number"), BindingRole("goal", "goal", "Number")]
    costs = {(role.identity, candidate.key): 0. for role in roles for candidate in ctx.referents}
    result = bind_context_roles(ctx, roles, costs=costs,
                                relations=[BindingRelation("source", "goal", "reaches")])
    assert result.bindings == (("goal", ("turn2", "literal")), ("source", ("turn1", "result")))


def test_soft_relational_energy_matches_exhaustive_search():
    roles = [BindingRole("left", "value", "Number"), BindingRole("right", "value", "Number")]
    keys = (("turn1", "result"), ("turn2", "literal"))
    relation = BindingRelation("left", "right", "different", violation_cost=5.)
    for a, b in product((0., 1., 3.), repeat=2):
        costs = {("left", keys[0]): 0., ("left", keys[1]): a,
                 ("right", keys[0]): 0., ("right", keys[1]): b}
        energies = sorted(costs["left", left] + costs["right", right]
                          + relation.cost(context(), left, right) for left, right in product(keys, repeat=2))
        result = bind_context_roles(context(), roles, costs=costs, relations=[relation])
        assert result.energy == pytest.approx(energies[0])
        assert result.margin == pytest.approx(energies[1] - energies[0])


@pytest.mark.parametrize("status", [1, 3, 4])
def test_solver_limits_are_not_infeasibility_or_success(monkeypatch, status):
    monkeypatch.setattr("scipy.optimize.milp", lambda *args, **kwargs: SimpleNamespace(status=status, x=None))
    with pytest.raises(ArgumentOptimizationIncompleteError):
        bind_context_roles(context(), [BindingRole("value", "theme", "Number",
                                                   referents=(("turn1", "result"),))])


def test_bounded_perturbation_uses_a_global_two_sided_gap():
    assert binding_margin_survives(1., [.1, .2])
    assert not binding_margin_survives(.6, [.1, .2])
    assert not binding_margin_survives(.5, [.1, .2])


def chart():
    return ScoredArgumentChart((
        (((5., 0, TokenSpan(0, 1)), (4., 1, TokenSpan(0, 1))),
         ((5., 1, TokenSpan(2, 3)), (4., 0, TokenSpan(2, 3)))),),
        2, RegisterUseContract(1, 1, 0, 1, True))


def test_chart_grounding_shares_the_existing_ssa_solver_and_preserves_original():
    original = chart()
    before = original.solve()
    roles = ((BindingRole("minuend", "minuend", "Number", referents=(("turn2", "literal"),)),
              BindingRole("subtrahend", "subtrahend", "Number", referents=(("turn1", "result"),))),)
    keys = {0: ("turn1", "result"), 1: ("turn2", "literal")}
    result = original.solve_grounded(context(), roles, keys)
    assert result.status == "bound" and result.assignment[1] == ((1, 0),)
    assert result.bindings == (("minuend", keys[1]), ("subtrahend", keys[0]))
    assert original.solve() == before


def test_chart_grounding_retains_joint_scores_and_measures_graph_ambiguity():
    roles = ((BindingRole("left", "minuend", "Number"), BindingRole("right", "subtrahend", "Number")),)
    keys = {0: ("turn1", "result"), 1: ("turn2", "literal")}
    result = chart().solve_grounded(context(), roles, keys,
        relations=[BindingRelation("left", "right", "different", violation_cost=10.)])
    assert result.assignment[1] == ((0, 1),) and result.margin == pytest.approx(2.)
    assert chart().solve_grounded(context(), roles, keys, minimum_margin=3.).status == "ambiguous"


def test_pair_factor_can_change_a_joint_chart_winner():
    from core.learning.semantic_argument_optimization import optimize_argument_chart

    original = chart()
    result = optimize_argument_chart(original.options, n_inputs=2, contract=original.contract,
        option_pair_factors=(((0, 0, 1), (0, 1, 1), 4.),))
    assert result[1] == ((1, 0),) and result[0] == pytest.approx(12.)
    forbidden = optimize_argument_chart(original.options, n_inputs=2, contract=original.contract,
        option_pair_factors=(((0, 0, 0), (0, 1, 0), None),))
    assert forbidden[1] == ((1, 0),)
