"""Nonlinear meaning, temporal custody and observed symbol acquisition contracts."""

import json
from dataclasses import replace
from fractions import Fraction

import mlx.core as mx
import pytest

from core.learning.semantic_context_binding import BindingContext, BindingRole, ContextReferent
from core.learning.semantic_diagram_workspace import (
    DiagramProposal,
    PublicDiagramCondition,
    replay_diagram_choice,
    retain_diagram_choice,
)
from core.learning.semantic_discourse_smoothing import DiscourseFactor, infer_discourse_references
from core.learning.semantic_grounded_binding_engine import (
    GroundedBindingEngine,
    GroundedBindingEvidence,
)
from core.learning.semantic_relational_pointer import RelationalBindingPointer
from core.learning.semantic_semasiographic import (
    DiagramEvent,
    DiagramObservations,
    DiagramPresentation,
    DiagramReferent,
    MeaningDiagram,
    MeaningFrame,
    diagram_event_tensor,
    diagram_role_contract,
    execute_meaning_diagram,
    resolve_meaning_diagram,
)
from core.learning.semantic_symbol_inquiry import (
    DemonstrationRequest,
    ObservedDemonstration,
    SymbolHypothesis,
    SymbolInquiry,
    propose_floor_symbol_meanings,
)
from core.learning.semantic_temporal_constraints import (
    TemporalConstraint,
    assess_temporal_constraints,
    temporal_custody_contract,
)


def subtraction(*, holes=False, at=0):
    a, b, e = ("page", "a"), ("page", "b"), ("page", "event")
    refs = (DiagramReferent(a, "integer", "scene:a", at), DiagramReferent(b, "integer", "scene:b", at))
    event = DiagramEvent(e, "unknown-ink", (("arc1", None if holes else a),
                                           ("arc2", None if holes else b)), "visible:ink", at)
    diagram = MeaningDiagram(refs, (event,))
    frame = MeaningFrame("unknown-ink", (("arc1", "integer"), ("arc2", "integer")), "integer", "sub")
    return diagram, {("actual", "unknown-ink"): frame}, {a: 10, b: 5}, e


def observations(diagram):
    return DiagramObservations(
        {e.key: mx.array([[1., 0., 0., 1.]]) for e in diagram.events},
        {(e.key, r): mx.array([[0., 1., 1., 0.]]) for e in diagram.events for r, _ in e.ports},
        {v.key: mx.array([[1., 0., 1., 0.]]) for v in (*diagram.referents, *diagram.events)})


def test_nonlinear_presentation_roundtrips_without_becoming_the_meaning():
    diagram, frames, inputs, output = subtraction()
    keys = tuple(v.key for v in (*diagram.referents, *diagram.events))
    left = DiagramPresentation(diagram, "image:first", tuple((k, (float(i), 0.)) for i, k in enumerate(keys)), keys)
    right = DiagramPresentation(diagram, "image:rotated", tuple((k, (0., float(-i))) for i, k in enumerate(keys)),
                                tuple(reversed(keys)), ("different", "spoken", "form"))
    for presentation in (left, right):
        payload = json.loads(json.dumps(presentation.to_dict()))
        restored = DiagramPresentation.from_dict(payload)
        assert restored.diagram == diagram and restored.layout == presentation.layout
        assert execute_meaning_diagram(restored.diagram, frames, inputs, output=output, cutoff=0).result == 5
    assert left.diagram.meaning_sha256 == right.diagram.meaning_sha256
    assert diagram_role_contract() == ()


def test_equal_values_do_not_hide_role_reversal_or_entity_identity():
    diagram, frames, inputs, output = subtraction()
    event = diagram.events[0]
    swapped = replace(diagram, events=(replace(event, ports=tuple((r, k) for (r, _), (_, k) in
        zip(event.ports, reversed(event.ports), strict=True))),))
    assert diagram.meaning_sha256 != swapped.meaning_sha256
    inputs = {key: 5 for key in inputs}
    assert execute_meaning_diagram(diagram, frames, inputs, output=output, cutoff=0).result == 0
    assert execute_meaning_diagram(swapped, frames, inputs, output=output, cutoff=0).result == 0
    mapping = {v.key: ("renamed", v.key[1]) for v in (*diagram.referents, *diagram.events)}
    renamed = diagram.rename(mapping)
    assert execute_meaning_diagram(renamed, frames, {mapping[k]: v for k, v in inputs.items()},
        output=mapping[output], cutoff=0).result == 0


def test_parallel_fork_join_is_lowered_by_dependencies_not_storage_or_reading_order():
    diagram, frames, inputs, output = subtraction()
    a, b = tuple(inputs)
    parallel, join = ("page", "parallel"), ("page", "join")
    events = (DiagramEvent(join, "combine", (("left", output), ("right", parallel)), "observed:join"),
              DiagramEvent(parallel, "combine", (("left", a), ("right", b)), "observed:parallel"),
              diagram.events[0])
    whole = replace(diagram, events=events)
    frames["actual", "combine"] = MeaningFrame("combine", (("left", "integer"), ("right", "integer")),
                                               "integer", "add")
    assert execute_meaning_diagram(whole, frames, inputs, output=join, cutoff=0).result == 20
    reversed_storage = replace(whole, events=tuple(reversed(events)))
    assert whole.meaning_sha256 == reversed_storage.meaning_sha256


@pytest.mark.parametrize("change", ["negated", "hypothetical", "cycle", "unknown", "future", "type"])
def test_formal_execution_does_not_create_source_truth_or_missing_semantics(change):
    diagram, frames, inputs, output = subtraction()
    e = diagram.events[0]
    if change == "negated":
        diagram = replace(diagram, events=(replace(e, polarity=False),))
    elif change == "hypothetical":
        diagram = replace(diagram, events=(replace(e, modality="hypothetical"),))
    elif change == "cycle":
        diagram = replace(diagram, events=(replace(e, ports=(("arc1", e.key), ("arc2", next(iter(inputs))))),))
    elif change == "unknown":
        frames = {}
    elif change == "future":
        diagram = replace(diagram, referents=tuple(replace(v, available_at=1) for v in diagram.referents))
    else:
        inputs[next(iter(inputs))] = (1, 2)
    with pytest.raises(ValueError):
        execute_meaning_diagram(diagram, frames, inputs, output=output, cutoff=0)


def test_uninterpreted_concepts_keep_roles_scope_and_modality_in_the_analogy_graph():
    diagram, _, _, _ = subtraction()
    event = replace(diagram.events[0], symbol="promise", modality="reported", world="story")
    diagram = replace(diagram, events=(event,))
    relations = {r.predicate for r in diagram.structure_graph().relations}
    assert {"predicate:promise", "role:arc1", "role:arc2", "mode:reported", "world:story"} <= relations


def test_interpreted_structure_normalizes_surface_symbols_and_retains_source_roles():
    diagram, frames, _, _ = subtraction()
    frame = next(iter(frames.values()))
    event = diagram.events[0]
    renamed = replace(diagram, events=(replace(event, symbol="different-word",
        ports=tuple(("new-" + r, k) for r, k in event.ports)),))
    renamed_frame = replace(frame, symbol="different-word", roles=tuple(("new-" + r, t) for r, t in frame.roles))
    assert diagram.structure_graph(frames=frames).relations == renamed.structure_graph(
        frames={("actual", "different-word"): renamed_frame}).relations
    future = replace(diagram, events=(replace(event, available_at=5),))
    assert not any(r.predicate.startswith("predicate:") for r in future.structure_graph(cutoff=4).relations)
    assert any(r.predicate == "predicate:sub" for r in future.structure_graph(cutoff=5, frames=frames).relations)
    diff = diagram.diff(renamed)
    assert diff["changes"]["events"][0]["key"] == event.key
    assert not diff["observed_world_change"]


def test_temporal_closure_reasons_backwards_without_reversing_causality_or_evidence_time():
    constraints = (TemporalConstraint("zero", "arrival", 10, 10, "arrival-record", 7),
                   TemporalConstraint("departure", "arrival", "5/2", "7/2", "travel-model", 2))
    early = assess_temporal_constraints(constraints, cutoff=2, points=("zero",))
    assert early.interval("zero", "departure") == (None, None)
    later = assess_temporal_constraints(constraints, cutoff=7)
    assert later.interval("zero", "departure") == (Fraction(13, 2), Fraction(15, 2))
    assert later.entails(TemporalConstraint("zero", "departure", 6, 8, "query"))
    assert not later.entails(TemporalConstraint("arrival", "departure", 1, None, "query"))
    assert temporal_custody_contract() == ()


def test_temporal_worlds_conflicts_and_parallel_partial_order():
    constraints = (TemporalConstraint("a", "b", 2, 3, "actual"),
                   TemporalConstraint("a", "b", -3, -2, "counterfactual", world="fiction"),
                   TemporalConstraint("a", "c", 2, 3, "parallel"))
    actual = assess_temporal_constraints(constraints, cutoff=0)
    assert actual.interval("b", "c") == (-1, 1)
    assert actual.excluded == ("counterfactual",)
    contradicted = assess_temporal_constraints((*constraints,
        TemporalConstraint("b", "a", 1, 2, "incompatible")), cutoff=0)
    assert not contradicted.consistent
    with pytest.raises(ValueError, match="inconsistent"):
        contradicted.interval("a", "b")


def test_global_revision_calls_the_actual_binder_and_keeps_all_alternatives():
    diagram, frames, inputs, output = subtraction(holes=True)
    engine = GroundedBindingEngine(RelationalBindingPointer(4, relation_width=4, rounds=0), evidence_weight=0.)
    # A public frame with holes is genuinely ambiguous; null costs do not invent a binding.
    resolved, binding, receipt = resolve_meaning_diagram(engine, diagram, frames, observations(diagram), cutoff=0)
    assert resolved is None and binding.status != "bound"
    assert len(receipt["binding"]["roles"]) == 2
    fixed, frames, inputs, output = subtraction()
    proposal = DiagramProposal("first", fixed, frames, observations(fixed), inputs, output)
    later_bad = replace(proposal, identity="second", diagram=replace(fixed,
        temporal=(TemporalConstraint("a", "b", 2, 2, "seen"),
                  TemporalConstraint("b", "a", 1, 1, "later", 1))))
    early = engine.revise_diagrams((proposal, later_bad), cutoff=0)
    assert early.status == "ambiguous" and early.selected is None
    later = engine.revise_diagrams((proposal, later_bad), cutoff=1, previous=early)
    assert later.selected == "first" and later.version == 2
    assert later.candidates[1]["status"] == "inadmissible"
    assert later.candidates[0]["predicted_result"] == 5


@pytest.mark.asyncio
async def test_commit_requires_the_exact_selected_snapshot_and_never_authorizes_action():
    diagram, frames, inputs, output = subtraction()
    engine = GroundedBindingEngine(RelationalBindingPointer(4, relation_width=4), evidence_weight=0.)
    state = engine.revise_diagrams((DiagramProposal("one", diagram, frames, observations(diagram), inputs, output),), cutoff=0)
    class Gateway:
        async def mutate(self, request):
            return request
    with pytest.raises(ValueError, match="stale"):
        await retain_diagram_choice(state, Gateway(), expected_identity="old")
    request = await retain_diagram_choice(state, Gateway(), expected_identity=state.identity)
    assert not request.new_value["action_authority"] and not request.new_value["semantic_truth_certified"]
    restored, execution = replay_diagram_choice(json.loads(json.dumps(request.new_value)))
    assert restored.identity == state.identity and execution.result == 5


@pytest.mark.asyncio
async def test_real_state_gateway_persists_the_graph_and_symbol_evidence_across_gateway_instances(tmp_path, monkeypatch):
    from core.runtime.receipts import ReceiptStore
    from core.state.state_gateway import ConcreteStateGateway

    store = ReceiptStore(root=tmp_path / "receipts")
    monkeypatch.setattr("core.state.state_gateway.get_receipt_store", lambda: store)
    gateway = ConcreteStateGateway(root=tmp_path / "state", governance_decide=lambda **_: True)
    diagram, frames, inputs, output = subtraction()
    inquiry = SymbolInquiry((SymbolHypothesis("meaning", tuple(frames.values()), "lesson"),))
    request = DemonstrationRequest(diagram, tuple(inputs.items()), output, 0, "lesson")
    inquiry.observe(ObservedDemonstration(request, 5, "scene", "observed:1", 0), cutoff=0)
    mutation = await inquiry.retain(gateway)
    engine = GroundedBindingEngine(RelationalBindingPointer(4), evidence_weight=0.)
    state = engine.revise_diagrams((DiagramProposal("one", diagram, frames, observations(diagram), inputs, output),), cutoff=0)
    graph_mutation = await retain_diagram_choice(state, gateway, expected_identity=state.identity)
    reopened = ConcreteStateGateway(root=tmp_path / "state")
    restored = await SymbolInquiry.load(reopened, mutation.key)
    assert restored.receipt() == inquiry.receipt()
    payload = await reopened.read(graph_mutation.key, domain="semantic_diagram_choices", fresh=True)
    restored_state, execution = replay_diagram_choice(payload)
    assert restored_state.identity == state.identity and execution.result == 5
    assert list((tmp_path / "state" / "semantic_symbol_meaning").glob("*.json"))
    denied = ConcreteStateGateway(root=tmp_path / "denied")
    with pytest.raises(PermissionError):
        await inquiry.retain(denied)


@pytest.mark.asyncio
async def test_graph_restore_reexecutes_instead_of_trusting_a_rehashed_result():
    from core.learning.semantic_semasiographic import digest

    diagram, frames, inputs, output = subtraction()
    engine = GroundedBindingEngine(RelationalBindingPointer(4), evidence_weight=0.)
    state = engine.revise_diagrams((DiagramProposal("one", diagram, frames, observations(diagram), inputs, output),), cutoff=0)
    class Gateway:
        async def mutate(self, request):
            return request
    mutation = await retain_diagram_choice(state, Gateway(), expected_identity=state.identity)
    payload = json.loads(json.dumps(mutation.new_value))
    row = payload["working_state"]["candidates"][0]
    row["predicted_result"] = 123
    workspace = payload["working_state"]
    workspace["content_sha256"] = digest({k: v for k, v in workspace.items() if k != "content_sha256"})
    payload["selected"] = row
    payload["state_identity"] = digest((state.version, state.cutoff, workspace["candidates"],
                                        state.selected, state.margin, state.status))
    payload["content_sha256"] = digest({k: v for k, v in payload.items() if k != "content_sha256"})
    with pytest.raises(ValueError, match="independent floor"):
        replay_diagram_choice(payload)


def test_symbol_acquisition_requires_new_observations_and_can_retract_bad_evidence():
    diagram, frames, inputs, output = subtraction()
    frame = next(iter(frames.values()))
    hypotheses = (SymbolHypothesis("forward", (frame,), "lesson"),
                  SymbolHypothesis("reverse", (replace(frame, roles=tuple(reversed(frame.roles))),), "lesson"))
    inquiry = SymbolInquiry(hypotheses)
    equal = DemonstrationRequest(diagram, tuple((k, 5) for k in inputs), output, 0, "lesson")
    discriminating = DemonstrationRequest(diagram, tuple(inputs.items()), output, 0, "lesson")
    experiment = inquiry.next_experiment((equal, discriminating))
    assert experiment.do == discriminating and experiment.settles > 0
    with pytest.raises(ValueError, match="future evidence"):
        inquiry.next_experiment((replace(discriminating, cutoff=1),))
    assert len(inquiry.remaining) == 2
    observation = ObservedDemonstration(discriminating, 5, "sensor", "independent:1", 1)
    with pytest.raises(ValueError, match="future"):
        inquiry.observe(observation, cutoff=0)
    receipt = inquiry.observe(observation, cutoff=1)
    assert receipt["retained"] == ["forward"] and not receipt["serving_authority"]
    assert inquiry.replay(cutoff=0)["status"] == "unsupported_survivors"
    assert inquiry.replay(cutoff=1, retracted=(observation.identity,))["status"] == "unsupported_survivors"


def test_unknown_symbols_and_contexts_stay_unknown_and_no_hypothesis_is_forced_to_win():
    diagram, frames, inputs, output = subtraction()
    frame = next(iter(frames.values()))
    inquiry = SymbolInquiry((SymbolHypothesis("wrong-context", (frame,), "elsewhere"),
                             SymbolHypothesis("covered", (frame,), "lesson")))
    request = DemonstrationRequest(diagram, tuple(inputs.items()), output, 0, "lesson")
    receipt = inquiry.observe(ObservedDemonstration(request, 123, "sensor", "new", 0), cutoff=0)
    assert receipt["retained"] == ["wrong-context"]  # silence is not supporting evidence
    assert receipt["status"] == "unsupported_survivors" and receipt["observed_agreements"] == {"wrong-context": 0}
    assert not receipt["hypothesis_space_complete"]
    only = SymbolInquiry((SymbolHypothesis("covered", (frame,), "lesson"),))
    assert only.observe(ObservedDemonstration(request, 123, "sensor", "new", 0), cutoff=0)["status"] == "contradicted"


def test_whole_event_tensor_preserves_reentrant_identity_and_equal_fillers():
    import numpy as np

    diagram, _, inputs, output = subtraction()
    states = {key: (1., 2., 3.) for key in inputs}
    tensor = diagram_event_tensor(diagram, output, states)
    assert np.array_equal(tensor.unbind("arc1"), states[dict(diagram.events[0].ports)["arc1"]])
    assert len(set(tensor.source_keys.values())) == 2
    event = replace(diagram.events[0], ports=tuple((r, next(iter(inputs))) for r, _ in diagram.events[0].ports))
    tensor = diagram_event_tensor(replace(diagram, events=(event,)), output, states)
    assert len(set(tensor.source_keys.values())) == 1


def test_source_bound_smoothing_uses_later_context_only_after_it_is_available():
    import numpy as np

    keys = (("discourse", "alice"), ("discourse", "bob"))
    candidates = (keys, keys)
    factors = (DiscourseFactor((0, 1), np.log([[.9, .1], [.1, .9]]), "continuity", 0),
               DiscourseFactor((1,), np.log([.01, .99]), "later_referent_observation", 1))
    early = infer_discourse_references(candidates, factors, cutoff=0)
    filtered = infer_discourse_references(candidates, factors, cutoff=1, mode="filter")
    smoothed = infer_discourse_references(candidates, factors, cutoff=1)
    assert early.marginals[0][keys[1]] == pytest.approx(.5)
    assert filtered.marginals[0][keys[1]] == pytest.approx(.5)
    assert smoothed.marginals[0][keys[1]] == pytest.approx(.892)
    engine = GroundedBindingEngine(RelationalBindingPointer(4), evidence_weight=0.)
    costs, posterior = engine.discourse_costs(candidates, factors, {"old_mention": 0}, cutoff=1)
    assert costs["old_mention", keys[1]] < costs["old_mention", keys[0]]
    assert posterior.mode == "smooth"
    # Candidate renaming changes identities, not probability mass.
    renamed = tuple(tuple(("renamed", key[1]) for key in row) for row in candidates)
    assert infer_discourse_references(renamed, factors, cutoff=1).marginals[0][renamed[0][1]] == pytest.approx(.892)


def test_smoothing_rejects_explosive_or_inconsistent_factor_models():
    import numpy as np

    keys = (("d", "a"), ("d", "b"))
    with pytest.raises(ValueError, match="exact bound"):
        infer_discourse_references((keys,) * 17, (), cutoff=0)
    zero = DiscourseFactor((0,), np.array([0., -np.inf]), "a_only", 0)
    opposite = DiscourseFactor((0,), np.array([-np.inf, 0.]), "b_only", 0)
    with pytest.raises(ValueError, match="every assignment"):
        infer_discourse_references((keys,), (zero, opposite), cutoff=0)


def test_goal_boundary_is_a_public_constraint_not_future_observation():
    diagram, frames, inputs, output = subtraction()
    frame = next(iter(frames.values()))
    reverse_frames = {("actual", frame.symbol): replace(frame, roles=tuple(reversed(frame.roles)))}
    condition = PublicDiagramCondition(5, 5, "user_declared_goal", 1, "public_goal")
    engine = GroundedBindingEngine(RelationalBindingPointer(4), evidence_weight=0.)
    first = DiagramProposal("forward", diagram, frames, observations(diagram), inputs, output, conditions=(condition,))
    second = replace(first, identity="reverse", frames=reverse_frames)
    early = engine.revise_diagrams((first, second), cutoff=0)
    assert early.status == "ambiguous"
    later = engine.revise_diagrams((first, second), cutoff=1, previous=early)
    assert later.selected == "forward" and later.candidates[1]["status"] == "infeasible"
    assert later.candidates[0]["conditions"] == [("user_declared_goal", "public_goal")]


def test_endpoint_constraints_resolve_holes_inside_joint_binding_not_after_a_local_winner():
    diagram, frames, inputs, output = subtraction(holes=True)
    condition = PublicDiagramCondition(5, 5, "observed_endpoint", 1, "observed")
    engine = GroundedBindingEngine(RelationalBindingPointer(4), evidence_weight=0.)
    proposal = DiagramProposal("unknown_bindings", diagram, frames, observations(diagram), inputs, output,
                               conditions=(condition,))
    early = engine.revise_diagrams((proposal,), cutoff=0)
    assert early.status == "unresolved" and early.candidates[0]["status"] == "ambiguous"
    later = engine.revise_diagrams((proposal,), cutoff=1, previous=early)
    assert later.selected == proposal.identity
    resolved = MeaningDiagram.from_dict(later.candidates[0]["diagram"])
    assert dict(resolved.events[0].ports) == {"arc1": ("page", "a"), "arc2": ("page", "b")}
    assert later.candidates[0]["predicted_result"] == 5


@pytest.mark.asyncio
async def test_symbol_retention_reloads_full_observations_retractions_and_predictions():
    diagram, frames, inputs, output = subtraction()
    frame = next(iter(frames.values()))
    inquiry = SymbolInquiry((SymbolHypothesis("forward", (frame,), "lesson"),
        SymbolHypothesis("reverse", (replace(frame, roles=tuple(reversed(frame.roles))),), "lesson")))
    request = DemonstrationRequest(diagram, tuple(inputs.items()), output, 0, "lesson")
    observation = ObservedDemonstration(request, 5, "sensor", "independent:1", 1)
    inquiry.observe(observation, cutoff=1)
    class Gateway:
        async def mutate(self, request):
            return request
    stored = await inquiry.retain(Gateway())
    payload = json.loads(json.dumps(stored.new_value))
    restored = SymbolInquiry.restore(payload)
    assert restored.receipt() == inquiry.receipt()
    assert restored.observations[0].request.diagram == diagram
    assert restored.replay(cutoff=1, retracted=(observation.identity,))["retained"] == ["forward", "reverse"]
    # Repeated observation sources are idempotent and do not undo a retraction.
    assert restored.observe(observation, cutoff=2)["retracted"] == (observation.identity,)
    assert SymbolInquiry.restore(json.loads(json.dumps(restored.receipt()))).receipt() == restored.receipt()


def test_symbol_restore_independently_checks_conclusions_even_with_a_recomputed_hash():
    from core.learning.semantic_semasiographic import digest

    diagram, frames, inputs, output = subtraction()
    inquiry = SymbolInquiry((SymbolHypothesis("meaning", tuple(frames.values()), "lesson"),))
    payload = json.loads(json.dumps(inquiry.receipt()))
    payload["retained"] = []
    body = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    payload["receipt_sha256"] = digest(body)
    with pytest.raises(ValueError, match="evidence replay"):
        SymbolInquiry.restore(payload)
    payload = json.loads(json.dumps(inquiry.receipt()))
    payload["cutoff"] = 9
    with pytest.raises(ValueError, match="custody"):
        SymbolInquiry.restore(payload)


@pytest.mark.asyncio
async def test_symbol_load_binds_the_replayed_receipt_to_the_requested_storage_key():
    _, frames, _, _ = subtraction()
    inquiry = SymbolInquiry((SymbolHypothesis("meaning", tuple(frames.values()), "lesson"),))
    payload = inquiry.receipt()

    class Gateway:
        async def read(self, key, *, domain, fresh):
            assert domain == "semantic_symbol_meaning" and fresh
            return payload

    assert (await SymbolInquiry.load(Gateway(), payload["receipt_sha256"])).receipt() == payload
    with pytest.raises(ValueError, match="storage key"):
        await SymbolInquiry.load(Gateway(), "different-receipt")


def test_symbol_uncertainty_accumulates_evidence_and_new_meanings_replay_old_observations():
    diagram, frames, inputs, output = subtraction()
    frame = next(iter(frames.values()))
    forward = SymbolHypothesis("forward", (frame,), "lesson")
    reverse = SymbolHypothesis("reverse", (replace(frame, roles=tuple(reversed(frame.roles))),), "lesson")
    inquiry = SymbolInquiry((forward, reverse), prior_weights={"forward": 1., "reverse": 3.},
                            mismatch_likelihood=.1)
    request = DemonstrationRequest(diagram, tuple(inputs.items()), output, 0, "lesson")
    receipt = inquiry.observe(ObservedDemonstration(request, 5, "sensor", "first", 1), cutoff=1)
    assert receipt["posterior"]["forward"] == pytest.approx(1 / 1.3)
    assert receipt["posterior"]["reverse"] == pytest.approx(.3 / 1.3)
    receipt = inquiry.observe(ObservedDemonstration(request, 5, "sensor", "second", 2), cutoff=2)
    assert receipt["posterior"]["forward"] == pytest.approx(1 / 1.03)
    assert not receipt["posterior_is_calibrated_accuracy"]
    assert SymbolInquiry.restore(json.loads(json.dumps(receipt))).receipt() == receipt
    exhausted = SymbolInquiry((reverse,))
    assert exhausted.observe(ObservedDemonstration(request, 5, "sensor", "first", 1), cutoff=1)["status"] == "contradicted"
    assert exhausted.extend((forward,))["retained"] == ["forward"]


def test_floor_symbol_proposals_include_both_directions_and_preserve_indistinguishable_meanings():
    hypotheses = propose_floor_symbol_meanings("ink", (("arc1", "integer"), ("arc2", "integer")),
                                               "integer", "lesson")
    sub_roles = {h.frames[0].roles for h in hypotheses if h.frames[0].operation == "sub"}
    assert sub_roles == {(("arc1", "integer"), ("arc2", "integer")),
                         (("arc2", "integer"), ("arc1", "integer"))}
    assert len(hypotheses) == 10
    with pytest.raises(ValueError, match="exact bound"):
        propose_floor_symbol_meanings("ink", (("a", "integer"), ("b", "integer")),
                                     "integer", "lesson", limit=1)


@pytest.mark.asyncio
async def test_observation_to_symbol_revision_to_grounded_graph_to_retained_floor_replay_is_one_path():
    diagram, _, inputs, output = subtraction()
    inquiry = SymbolInquiry(propose_floor_symbol_meanings("unknown-ink",
        (("arc1", "integer"), ("arc2", "integer")), "integer", "lesson"))
    first = DemonstrationRequest(diagram, tuple(inputs.items()), output, 0, "lesson")
    assert inquiry.observe(ObservedDemonstration(first, 5, "scene", "first", 1), cutoff=1)["status"] == "ambiguous"
    equal = replace(first, inputs=tuple((k, 5) for k in inputs))
    different = replace(first, inputs=((('page', 'a'), 13), (('page', 'b'), 5)))
    assert inquiry.next_experiment((equal, different)).do == different
    receipt = inquiry.observe(ObservedDemonstration(different, 8, "scene", "second", 2), cutoff=2)
    assert receipt["status"] == "one_surviving_proposal"
    learned = next(iter(inquiry.remaining.values()))
    assert learned.frames[0].operation == "sub"
    class Gateway:
        async def mutate(self, request):
            return request
    saved = await inquiry.retain(Gateway())
    restored = SymbolInquiry.restore(json.loads(json.dumps(saved.new_value)))
    learned = next(iter(restored.remaining.values()))
    engine = GroundedBindingEngine(RelationalBindingPointer(4), evidence_weight=0.)
    proposal = DiagramProposal("learned", diagram, learned.interpretations, observations(diagram),
                               dict(different.inputs), output)
    state = engine.revise_diagrams((proposal,), cutoff=2)
    assert state.selected == "learned"
    graph = await retain_diagram_choice(state, Gateway(), expected_identity=state.identity)
    _restored_state, execution = replay_diagram_choice(json.loads(json.dumps(graph.new_value)))
    assert execution.result == 8
    assert not graph.new_value["semantic_truth_certified"]


def test_smoothing_keeps_extremely_unlikely_distinct_from_impossible_bindings():
    import numpy as np

    keys = (("d", "a"), ("d", "b"))
    posterior = infer_discourse_references((keys,), (DiscourseFactor((0,), [0., -1000.], "rare", 0),), cutoff=0)
    assert posterior.marginals[0][keys[1]] == 0.  # display underflow is not a hard exclusion
    assert posterior.binding_costs({"mention": 0})["mention", keys[1]] == pytest.approx(1000.)
    exact = infer_discourse_references((keys,), (DiscourseFactor((0,), [0., -np.inf], "impossible", 0),), cutoff=0)
    assert ("mention", keys[1]) not in exact.binding_costs({"mention": 0})


def test_smoothing_is_causally_used_by_the_actual_grounded_binder_with_a_zero_weight_lesion():
    import numpy as np

    keys = (("d", "a"), ("d", "b"))
    records = tuple(ContextReferent(*key, "Number", "source:" + key[1]) for key in keys)
    role = BindingRole("mention", "value", "Number")
    evidence = GroundedBindingEvidence("discourse:1", BindingContext(records, {"Number": None}), (role,),
        {role.identity: mx.array([[1., 0., 0., 1.]])}, {role.identity: mx.array([[0., 1., 1., 0.]])},
        {key: mx.array([[1., 0., 1., 0.]]) for key in keys})
    engine = GroundedBindingEngine(RelationalBindingPointer(4), evidence_weight=0.)
    factors = (DiscourseFactor((0,), [0., -np.inf], "available_source", 1),)
    early, _ = engine.resolve_discourse(evidence, (keys,), factors, {"mention": 0}, cutoff=0)
    late, receipt = engine.resolve_discourse(evidence, (keys,), factors, {"mention": 0}, cutoff=1)
    lesion, _ = engine.resolve_discourse(evidence, (keys,), factors, {"mention": 0}, cutoff=1, weight=0.)
    assert early.status == lesion.status == "ambiguous"
    assert late.executable and late.bindings == (("mention", keys[0]),)
    assert receipt["evidence"] == ("available_source",) and not receipt["serving_authority"]
    with pytest.raises(ValueError, match="source identities"):
        engine.resolve_discourse(evidence, ((keys[0],),), (), {"mention": 0}, cutoff=1)
