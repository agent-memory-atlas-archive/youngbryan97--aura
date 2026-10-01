"""All observed chart options reach the trained pointer and original solver."""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import pytest

from core.learning.semantic_argument_chart import ScoredArgumentChart
from core.learning.semantic_grounded_binding_engine import GroundedBindingEngine
from core.learning.semantic_grounded_chart_bridge import GroundedBindingChartSolver
from core.learning.semantic_program_ir import TokenSpan
from core.learning.semantic_program_transducer_fitting import RegisterUseContract
from core.learning.semantic_relational_pointer import (
    RelationalBindingPointer,
    pointer_choice_loss,
    semantic_role_features,
)


def fixture():
    mx.random.seed(33)
    states = mx.array([[[0., 1., 0., 0.]], [[1., 0., 0., 0.]],
                       [[1., 1., 1., 0.]], [[1., 0., 1., 0.]], [[0., 1., 0., 1.]]])
    pointer = RelationalBindingPointer(4, depths=1, relation_width=8, rounds=0)
    operations = mx.stack([states[2], states[2]])
    mentions, candidates = states[3:], states[:2]
    from core.learning.semantic_context_binding import BindingRole
    queries = semantic_role_features(tuple(BindingRole(str(slot), f"sub:operand:{slot}", "integer")
                                          for slot in range(2)))
    optimizer = optim.Adam(learning_rate=.03)
    for _ in range(24):
        loss, grads = nn.value_and_grad(pointer, lambda owner:
            pointer_choice_loss(owner(operations, mentions, candidates, role_features=queries), ((1,), (0,))))(pointer)
        optimizer.update(pointer, grads)
        mx.eval(pointer.parameters(), loss)
    chart = ScoredArgumentChart(((((5., 0, TokenSpan(3, 4)), (0., 1, TokenSpan(3, 4))),
                                  ((5., 1, TokenSpan(4, 5)), (0., 0, TokenSpan(4, 5)))),),
                                2, RegisterUseContract(1, 1, 0, 1, True))
    bridge = GroundedBindingChartSolver(GroundedBindingEngine(pointer, evidence_weight=100.), "public-source", states)
    arguments = dict(operation_nodes=(SimpleNamespace(operation="sub", span=TokenSpan(2, 3)),),
                     input_spans=(TokenSpan(0, 1), TokenSpan(1, 2)), inputs=(4, 32), source_id="public-source")
    return bridge, chart, arguments


def test_observed_pointer_can_override_wrong_baseline_without_pruning_any_option():
    bridge, chart, arguments = fixture()
    assert chart.solve()[1] == ((0, 1),)
    selected = bridge(chart, **arguments)
    assert selected[1] == ((1, 0),)
    assert bridge.last_resolution["all_options_retained"] and bridge.last_resolution["status"] == "bound"
    edges = bridge.last_resolution["edge_evidence"]
    assert len(edges) == 4 and {edge["role_id"] for edge in edges} == {"sub:operand:0", "sub:operand:1"}
    assert all(edge["combined_score"] == edge["baseline_score"] + 100 * edge["learned_relation_score"] for edge in edges)
    assert chart.solve()[1] == ((0, 1),)


def test_bridge_rejects_different_source_and_does_not_certify_an_energy_tie():
    bridge, chart, arguments = fixture()
    with pytest.raises(ValueError, match="different public source"):
        bridge(chart, **{**arguments, "source_id": "other-source"})
    bridge.engine.evidence_weight = 0.
    tied = ScoredArgumentChart(((((0., 0, TokenSpan(3, 4)), (0., 1, TokenSpan(3, 4))),
                                 ((0., 1, TokenSpan(4, 5)), (0., 0, TokenSpan(4, 5)))),),
                               2, chart.contract)
    assert bridge(tied, **arguments) is None
    assert bridge.last_resolution["status"] == "ambiguous"


def test_native_grammar_bridge_consumes_public_operation_role_and_all_registered_candidates():
    from dataclasses import replace

    from core.learning.semantic_context_binding import BindingRole
    from core.learning.semantic_grounded_chart_bridge import GroundedNativeReferenceScorer
    from core.learning.semantic_native_grammar import NativeGrammarDecision
    from tests.test_semantic_grounded_binding_engine import source

    evidence = source("public").evidence
    role = BindingRole("request", "sub:operand:1", "Number")
    evidence = replace(evidence, roles=(role,), operations={role.identity: evidence.operations["operand"]},
                       mentions={role.identity: evidence.mentions["operand"]})
    # The typed role must agree with the primitive contract.
    evidence = replace(evidence, roles=(replace(role, type_name="integer"),),
        context=replace(evidence.context, referents=tuple(replace(record, type_name="integer")
                       for record in evidence.context.referents), type_parents={"integer": None}))
    seen = []
    def read(**kwargs):
        seen.append(kwargs)
        return evidence, {index: record.key for index, record in enumerate(evidence.context.referents)}
    scorer = GroundedNativeReferenceScorer(GroundedBindingEngine(RelationalBindingPointer(4)),
        lambda decisions: tuple(float(index) for index in range(len(decisions))), read)
    choices = tuple(NativeGrammarDecision("unused-output", (0, 1), index, 0, 1, "sub") for index in range(2))
    assert scorer(choices) == (0., 1.)
    assert seen == [{"operation_name": "sub", "step_index": 0, "role_index": 1, "admitted_registers": (0, 1)}]
    assert scorer.receipts[0]["role_id"] == "sub:operand:1"
    assert not scorer.receipts[0]["labels_available_to_scorer"]
    with pytest.raises(ValueError, match="mix role"):
        scorer((choices[0], replace(choices[1], role_index=0)))
    assert scorer((NativeGrammarDecision("op", (0, 1), "sub"),)) == (0.,)


def test_ordinary_decode_reaches_grounded_bridge_without_teacher_argument_targets():
    from core.learning.semantic_grounded_binding_acquisition import (
        grounded_supervision_from_source_example,
    )
    from core.learning.semantic_grounded_binding_engine import grounded_source_loss
    from core.learning.semantic_program_compositional_transducer import (
        fit_compositional_semantic_program_transducer,
    )
    from tests.test_semantic_program_shared_transducer import _examples, _grounding

    examples = _examples()
    parent = fit_compositional_semantic_program_transducer(examples, input_grounding=_grounding())
    candidate = parent.with_global_constraint_arguments().with_joint_operation_argument_scores()
    item = next(example for example in examples if example.split == "test")
    width = item.hidden_channel_widths[0]
    # These are fixture-observed channels. They are not native transfer proof.
    states = mx.array(item.hidden_states.reshape(len(item.hidden_states), 3, width))
    mx.random.seed(91)
    pointer = RelationalBindingPointer(width, depths=3, relation_width=16, rounds=0)
    sources = [grounded_supervision_from_source_example(example,
        channels=example.hidden_channels) for example in examples if example.split == "train"]
    optimizer = optim.Adam(learning_rate=.01)
    for step in range(64):
        current = sources[step % len(sources)]
        loss, grads = nn.value_and_grad(pointer, lambda owner, current=current:
            grounded_source_loss(owner, current))(pointer)
        optimizer.update(pointer, grads)
        mx.eval(pointer.parameters(), loss)
    engine = GroundedBindingEngine(pointer, evidence_weight=.5)
    bridge = GroundedBindingChartSolver(engine, item.ir.source_text_sha256, states)
    calls = []

    def invoke(chart, **arguments):
        calls.append(arguments)
        assert set(arguments) == {"operation_nodes", "input_spans", "inputs", "source_id", "time_limit_s"}
        return bridge(chart, **arguments)

    arguments = dict(source_token_ids=item.ir.source_token_ids, hidden_states=item.hidden_states,
        public_inputs=item.public_inputs, source_text_sha256=item.ir.source_text_sha256,
        model_basis_sha256=candidate.model_basis_sha256)
    baseline = candidate.decode(**arguments)
    decoded = candidate.decode(**arguments, binding_chart_solver=invoke)
    assert calls and bridge.last_resolution["all_options_retained"]
    assert baseline.ir is not None and decoded.ir is not None, (decoded.refusal, bridge.last_resolution)
    assert decoded.ir.to_program() == item.ir.to_program()
    assert baseline.ir.to_program() != item.ir.to_program()
    with pytest.raises(ValueError, match="complete global"):
        parent.decode(**arguments, binding_chart_solver=invoke)
    with pytest.raises(ValueError, match="joint operation"):
        parent.with_global_constraint_arguments().decode(**arguments, binding_chart_solver=invoke)


def test_native_bridge_preserves_the_same_contextual_evidence_as_the_solver():
    from dataclasses import replace

    from core.learning.semantic_context_binding import BindingRole
    from core.learning.semantic_grounded_chart_bridge import GroundedNativeReferenceScorer
    from core.learning.semantic_native_grammar import NativeGrammarDecision
    from tests.test_semantic_grounded_binding_engine import source

    evidence = source("public").evidence
    role = BindingRole("request", "sub:operand:0", "integer", embedding=(1., 0.))
    records = tuple(replace(record, type_name="integer", embedding=vector)
        for record, vector in zip(evidence.context.referents, ((1., 0.), (-1., 0.)), strict=True))
    evidence = replace(evidence, roles=(role,),
        context=replace(evidence.context, referents=records, type_parents={"integer": None}),
        operations={role.identity: evidence.operations["operand"]},
        mentions={role.identity: evidence.mentions["operand"]})
    engine = GroundedBindingEngine(RelationalBindingPointer(4), evidence_weight=0.)
    scorer = GroundedNativeReferenceScorer(engine, lambda decisions: (0.,) * len(decisions),
        lambda **_query: (evidence, {index: record.key for index, record in enumerate(records)}))
    choices = tuple(NativeGrammarDecision("unused-output", (0, 1), index, 0, 0, "sub") for index in range(2))
    assert scorer(choices) == (0., -2.)
    assert engine.resolve(evidence).bindings == ((role.identity, records[0].key),)
    assert scorer.receipts[-1]["contextual_costs"] == (0., 2.)
