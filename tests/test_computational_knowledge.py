"""Calculated evidence retains units, uncertainty, scope, and model assumptions."""

import asyncio
import time
from dataclasses import replace
from fractions import Fraction

import pytest

from core.reasoning.computation_sandbox import (
    Formulation,
    retrieve_formulation_gaps,
    run_formulation,
    solve_model_unknown,
)
from core.reasoning.computational_knowledge import (
    ComputationModel,
    ComputationRequest,
    KnowledgeContext,
    QuantityBounds,
    ScopedPremise,
    compute_knowledge,
    compute_knowledge_batch,
    verify_computed_knowledge,
)
from core.reasoning.computational_models import MODEL_CELLS
from core.reasoning.scoped_logical_computation import derive_scoped_fact


def test_scoped_computation_claim_has_an_executable_observation_and_limited_scope():
    from core.organism.claims_computational_knowledge import install_computational_knowledge_claims
    from core.organism.model_validation import Evidence, Outcome, RuntimeModel, ValidationSuite

    suite = ValidationSuite()
    install_computational_knowledge_claims(suite)
    model = RuntimeModel().declare("scoped_computational_knowledge")
    assert suite.tests()[0].run(model).score.outcome is Outcome.PASS
    claim = suite.claims()[0]
    assert claim.evidence is Evidence.MEASURED_SYNTHETIC
    assert "no autonomous source grounding" in claim.evidence_note


def test_valid_formula_registration_does_not_depend_on_dummy_denominator_values():
    model = ComputationModel("gap_ratio.v1", "mathematics",
        (("a", "count"), ("b", "count"), ("c", "count")), "count", "a/(b-c)", (), "rational algebra")
    context = KnowledgeContext("request", 0., tuple(ScopedPremise(
        name, "request", QuantityBounds.of(value), "fixture", name)
        for name, value in (("a", 8), ("b", 5), ("c", 3))))
    request = ComputationRequest("ratio", model, tuple((name, name) for name in ("a", "b", "c")))
    assert compute_knowledge(request, context).result == QuantityBounds.of(4)
    zero = replace(context, premises=tuple(replace(item, value=QuantityBounds.of(3))
        if item.identity == "b" else item for item in context.premises))
    with pytest.raises(ValueError, match="zero"):
        compute_knowledge(request, zero)


def test_formula_declarations_cannot_change_through_mutable_input_metadata():
    with pytest.raises(ValueError, match="immutable"):
        ComputationModel("mutable", "math", [("a", "count")], "count", "a", (), "test")


def prepare(name, quantities, *, scope="request", kind="given"):
    model = MODEL_CELLS[name + ".v1"]
    values = tuple(ScopedPremise(key, scope, QuantityBounds.of(value, unit=unit), "fixture", key, kind)
                   for (key, unit), value in zip(model.inputs, quantities, strict=True))
    guards = tuple(ScopedPremise(key, scope, True, "fixture", key, proposition=key) for key in model.assumptions)
    request = ComputationRequest("result", model, tuple((name, name) for name, _unit in model.inputs),
                                 tuple((name, name) for name in model.assumptions))
    return request, KnowledgeContext(scope, 10., values + guards)


@pytest.mark.parametrize("name,inputs,result", [
    ("arithmetic_sum", (2, 3), 5), ("arithmetic_difference", (2, 3), -1),
    ("rectangular_area", (2, 3), 6), ("squared_distance_2d", (3, 4), 25),
    ("average_speed_distance", (10, 2), 20),
    ("constant_acceleration_position", (1, 2, 3, 4), 33),
    ("newton_force", (3, -2), -6), ("kinetic_energy", (2, 3), 9),
    ("hydrostatic_pressure", (1000, "9.81", 2), 19620),
    ("buoyancy", (1000, "9.81", 2), 19620),
    ("continuity_speed", (10, 2), 5), ("reynolds_number", (1000, 2, 1, "0.001"), 2000000),
    ("sensible_heat", (2, 3, -4), -24), ("ohmic_current", (12, 3), 4),
    ("electrical_power", (12, 3), 36), ("pressure_force", (10, 2), 20),
    ("mechanical_work", (10, 2), 20), ("hydraulic_power", (10, 2), 20),
    ("density_mass", (10, 2), 20), ("ideal_gas_pressure", (2, 8, 300, 3), 1600),
    ("molar_concentration", (2, 4), "1/2"), ("dilution_concentration", (10, 2, 4), 5),
    ("michaelis_menten_rate", (10, 3, 2), 6),
])
def test_reviewed_cells_recover_known_answers_with_dimensions(name, inputs, result):
    request, context = prepare(name, inputs)
    measured = compute_knowledge(request, context)
    assert measured.result == QuantityBounds.of(result, unit=request.model.output_unit)
    assert measured.hard_constraint
    assert verify_computed_knowledge(measured, scope="request", now=10).result == measured.result
    assert measured.to_dict()["source_interpretation_proven"] is False


def test_batch_records_128_arithmetic_cells_and_measures_its_cost():
    request, context = prepare("arithmetic_sum", (2, 3))
    requests = tuple(replace(request, identity=f"cell:{index}") for index in range(128))
    start = time.perf_counter()
    rows = asyncio.run(compute_knowledge_batch(requests, context))
    assert len(MODEL_CELLS) == 23 and len(rows) == 128
    assert all(item.result.lower == 5 for item in rows)
    print(f"128 arithmetic cells: {time.perf_counter() - start:.6f} seconds")


def test_interval_estimate_is_a_range_and_never_a_new_exact_fact():
    request, context = prepare("average_speed_distance", (10, 2))
    velocity = replace(context.premises[0], value=QuantityBounds.of(9, 11, "m/s"), kind="estimate")
    context = replace(context, premises=(velocity, *context.premises[1:]))
    measured = compute_knowledge(request, context)
    assert measured.result == QuantityBounds.of(18, 22, "m") and not measured.hard_constraint


@pytest.mark.parametrize("kind", ["assumption", "estimate"])
def test_uncertain_model_applicability_never_hardens_by_computation(kind):
    request, context = prepare("hydrostatic_pressure", (1000, 10, 2))
    premises = list(context.premises)
    premises[-1] = replace(premises[-1], kind=kind)
    assert not compute_knowledge(request, replace(context, premises=tuple(premises))).hard_constraint


@pytest.mark.parametrize("defect", ["scope", "expiry", "future", "wrong_unit", "false_guard", "missing_guard", "contradiction"])
def test_unestablished_knowledge_cannot_be_used_as_a_proved_constraint(defect):
    request, context = prepare("hydrostatic_pressure", (1000, 10, 2))
    premises = list(context.premises)
    if defect == "scope":
        premises[0] = replace(premises[0], scope="other")
    elif defect == "expiry":
        premises[0] = replace(premises[0], valid_until=9.)
    elif defect == "future":
        premises[0] = replace(premises[0], valid_from=11.)
    elif defect == "wrong_unit":
        premises[0] = replace(premises[0], value=QuantityBounds.of(1000, unit="s"))
    elif defect == "false_guard":
        premises[-1] = replace(premises[-1], value=False)
    elif defect == "missing_guard":
        premises.pop()
    else:
        premises.append(replace(premises[-1], identity="opposing", value=False))
    with pytest.raises(ValueError):
        compute_knowledge(request, replace(context, premises=tuple(premises)))


def test_receipt_replay_catches_tampered_results_and_new_expiry():
    request, context = prepare("arithmetic_difference", (2, 3))
    context = replace(context, premises=(replace(context.premises[0], valid_until=11), context.premises[1]))
    result = compute_knowledge(request, context)
    with pytest.raises(ValueError, match="replay"):
        verify_computed_knowledge(replace(result, result=QuantityBounds.of(1)), scope="request", now=10)
    with pytest.raises(ValueError, match="stale"):
        verify_computed_knowledge(result, scope="request", now=12)


def test_rational_ranges_enclose_every_grid_product_without_rounding():
    a, b = QuantityBounds.of(-2, 3), QuantityBounds.of("1/3", "7/3")
    for x in (Fraction(-2), Fraction(0), Fraction(3)):
        for y in (Fraction(1, 3), Fraction(1), Fraction(7, 3)):
            assert (a*b).lower <= x*y <= (a*b).upper
    assert (a**2).lower == 0 and (a**2).upper == 9
    with pytest.raises(ValueError, match="zero"):
        _ = b/a


@pytest.mark.parametrize("expression", ["__import__('os')", "a.real", "a[0]", "a**100", "a//b", "True"])
def test_models_are_bounded_arithmetic_data_not_executable_python(expression):
    with pytest.raises(ValueError):
        ComputationModel("untrusted", "math", (("a", "count"), ("b", "count")), "count", expression, (), "test")


def cross_domain():
    pressure, context = prepare("hydrostatic_pressure", (1000, 10, 2))
    pressure = replace(pressure, identity="pressure")
    force = ComputationRequest("force", MODEL_CELLS["pressure_force.v1"], (("pressure", "pressure"), ("area", "area")),
                               (("uniform_normal_pressure", "uniform_normal_pressure"),))
    work = ComputationRequest("work", MODEL_CELLS["mechanical_work.v1"], (("force", "force"), ("distance", "distance")),
                              (("constant_force_parallel_to_displacement", "constant_force_parallel_to_displacement"),))
    extras = (ScopedPremise("area", "request", QuantityBounds.of("1/10", unit="m^2"), "measurement", "area:1"),
              ScopedPremise("distance", "request", QuantityBounds.of(3, unit="m"), "measurement", "distance:1"),
              ScopedPremise("uniform_normal_pressure", "request", True, "fixture", "applicability:1", proposition="uniform_normal_pressure"),
              ScopedPremise("constant_force_parallel_to_displacement", "request", True, "fixture", "applicability:2",
                            proposition="constant_force_parallel_to_displacement"))
    recipe = Formulation("water_to_work", "Work from a hydrostatic pressure acting on a piston over a displacement.",
                         (work, force, pressure), ("work",))
    return recipe, replace(context, premises=context.premises + extras)


def test_unordered_cross_domain_cells_solve_as_one_graph_and_keep_ancestor_freshness():
    recipe, context = cross_domain()
    context = replace(context, premises=(replace(context.premises[0], valid_until=11), *context.premises[1:]))
    outcome = run_formulation(recipe, context)
    assert not outcome.gaps and outcome.to_dict()["all_consequences_hard"]
    assert [item.request.identity for item in outcome.completed] == ["pressure", "force", "work"]
    work = outcome.completed[-1]
    assert work.result == QuantityBounds.of(6000, unit="J") and "rho" in work.premise_ids
    with pytest.raises(ValueError, match="stale"):
        verify_computed_knowledge(work, scope="request", now=12)


def test_cross_domain_estimate_remains_uncertain_at_every_downstream_stage():
    recipe, context = cross_domain()
    context = replace(context, premises=(replace(context.premises[0], kind="estimate"), *context.premises[1:]))
    outcome = run_formulation(recipe, context)
    assert not outcome.gaps and all(not item.hard_constraint for item in outcome.completed)


def test_missing_pieces_produce_retrieval_queries_not_invented_values():
    recipe, context = cross_domain()
    context = replace(context, premises=tuple(item for item in context.premises if item.identity != "rho"))
    outcome = run_formulation(recipe, context)
    assert not outcome.completed and len(outcome.gaps) == 3
    retrieved = asyncio.run(retrieve_formulation_gaps(outcome, {"local_corpus": lambda query: [{"text": query}]}))
    assert len(retrieved) == 3 and all(row["premises_admitted"] is False for row in retrieved)


def test_reusable_recipe_round_trips_without_caching_an_answer_or_permitting_model_drift():
    recipe, context = cross_domain()
    payload = recipe.to_dict()
    restored = Formulation.from_dict(payload, MODEL_CELLS)
    assert restored == recipe and payload["retained_answers_available"] is False
    changed = replace(context, premises=(replace(context.premises[0], value=QuantityBounds.of(500, unit="kg/m^3")),
                                        *context.premises[1:]))
    assert run_formulation(restored, changed).completed[-1].result == QuantityBounds.of(3000, unit="J")
    payload["definition"] = "unsupported new meaning"
    with pytest.raises(ValueError, match="content differs"):
        Formulation.from_dict(payload, MODEL_CELLS)


def test_inverse_model_solves_missing_piece_and_rechecks_the_original_equation():
    model = MODEL_CELLS["ohmic_current.v1"]
    result, receipt = solve_model_unknown(model, unknown="resistance",
        known={"voltage": QuantityBounds.of(12, unit="V")}, output=QuantityBounds.of(4, unit="A"))
    assert result == QuantityBounds.of(3, unit="ohm") and receipt["exact_substitution_verified"]
    assert receipt["hard_constraint"] is False and receipt["unestablished_assumptions"]
    with pytest.raises(ValueError):
        solve_model_unknown(model, unknown="resistance", known={"voltage": QuantityBounds.of(0, unit="V")},
                            output=QuantityBounds.of(0, unit="A"))


def test_logic_cell_uses_independent_kernel_and_cannot_explode_from_contradiction():
    context = KnowledgeContext("request", 10., (
        ScopedPremise("fact", "request", True, "observation", "sensor:1", proposition="A", valid_until=11),
        ScopedPremise("rule", "request", True, "reviewed_rule", "rule:1", proposition="A -> B"),
    ))
    derived, receipt = derive_scoped_fact(context, ("fact", "rule"), identity="consequence", goal="B")
    assert derived.value is True and receipt["kernel_verdict"]["verified"]
    assert set(derived.dependencies) == {"fact", "rule"}
    contradiction = replace(context, premises=(*context.premises,
        ScopedPremise("negative", "request", False, "observation", "sensor:2", proposition="A")))
    with pytest.raises(ValueError, match="contradictory"):
        derive_scoped_fact(contradiction, ("fact", "negative"), identity="anything", goal="C")
    unknown, receipt = derive_scoped_fact(context, ("fact",), identity="unproved", goal="B")
    assert unknown is None and receipt["provable"] is False


def test_checked_logic_can_supply_a_model_guard_without_losing_hypothetical_status():
    request, context = prepare("average_speed_distance", (10, 2))
    facts = (ScopedPremise("constant", "request", True, "theory", "hypothesis:1", "assumption",
                           "constant_speed"),
             ScopedPremise("rule", "request", True, "definition", "rule:1",
                           proposition="constant_speed -> average_speed_over_interval"))
    context = replace(context, premises=(*context.premises[:2], *facts))
    fact, receipt = derive_scoped_fact(context, ("constant", "rule"),
        identity="average_speed_over_interval", goal="average_speed_over_interval")
    assert not receipt["hard_constraint"]
    context = replace(context, premises=(*context.premises, fact))
    result = compute_knowledge(request, context)
    assert result.result == QuantityBounds.of(20, unit="m") and not result.hard_constraint
    assert set(result.premise_ids) == {"v", "t", "constant", "rule", "average_speed_over_interval"}
