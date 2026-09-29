"""The small transfer check must test meaning invariance and binding sensitivity."""

import ast
import hashlib
import inspect
from copy import deepcopy

import pytest

from tools.evaluate_semantic_native_grammar import grammar_examples, source_input_types
from tools.semantic_native_relation_transfer import (
    adjudicate_native_relation_transfer,
    build_native_relation_transfer,
)
from tools.verify_semantic_native_grammar import verified_dataset, verified_examples

SEED = 2147438394


def test_all_native_grammar_arms_require_exclusive_non_evicting_lane_ownership():
    from tools.evaluate_semantic_native_grammar import main

    calls = [node for node in ast.walk(ast.parse(inspect.getsource(main)))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "standalone_model_lane"]
    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.literal_eval(keywords["require_exclusive"]) is True
    assert ast.literal_eval(keywords["allow_owner_eviction"]) is False
    assert ast.literal_eval(keywords["preemptible"]) is False


def identity(example):
    return hashlib.sha256(example.source_text.encode()).hexdigest()


def test_controls_preserve_values_but_separate_expression_role_and_dependency():
    from core.learning.semantic_native_decision_supervision import native_teacher_decisions
    from core.learning.semantic_native_relative_program import REGISTER_ENCODING

    cases = build_native_relation_transfer(seed=SEED)
    assert len(cases) == 9
    references = grammar_examples(dataset="natural_request", seed=SEED, count=3)
    assert [identity(case.reference) for case in cases[::3]] == [identity(item) for item in references]
    for case in cases:
        public, types = source_input_types(case.controlled.source_text)
        assert public.values == case.reference.inputs == case.controlled.inputs
        assert native_teacher_decisions(case.controlled.program, input_types=types,
                                       register_encoding=REGISTER_ENCODING)
        if case.kind == "paraphrase":
            assert case.reference.program == case.controlled.program
            assert case.reference.source_text != case.controlled.source_text
        else:
            before = case.reference.program.instructions
            after = case.controlled.program.instructions
            assert [item.op for item in before] == [item.op for item in after]
            differences = [(a.args, b.args) for a, b in zip(before, after, strict=True)
                           if a.args != b.args]
            assert len(differences) == 1
            original, changed = differences[0]
            if case.kind == "role":
                assert changed == original[::-1]
            else:
                assert sum(a != b for a, b in zip(original, changed, strict=True)) == 1
            assert case.reference.program.run(public.values) != case.controlled.program.run(public.values)


def test_evaluation_and_verification_rebuild_the_same_full_control_inventory():
    examples = grammar_examples(dataset="relation_transfer_controls", seed=SEED, count=9)
    plan = {"schema": "aura.semantic_native_grammar_plan.v11",
            "dataset": "relation_transfer_controls", "seed": SEED,
            "sources": [identity(example) for example in examples]}
    report = {"dataset": plan["dataset"], "seed": SEED}
    assert verified_dataset(plan, report) == (plan["dataset"], SEED)
    assert verified_examples(plan, dataset=plan["dataset"], seed=SEED) == examples
    with pytest.raises(ValueError, match="all nine"):
        grammar_examples(dataset=plan["dataset"], seed=SEED, count=3)
    with pytest.raises(ValueError, match="complete controlled inventory"):
        verified_examples({**plan, "sources": plan["sources"][:3]}, dataset=plan["dataset"], seed=SEED)
    with pytest.raises(ValueError, match="dataset"):
        verified_dataset({**plan, "schema": "aura.semantic_native_grammar_plan.v3"}, report)


def fixture():
    cases = build_native_relation_transfer(seed=SEED)
    references = grammar_examples(dataset="natural_request", seed=SEED, count=3)
    base = {"schema": "aura.semantic_native_grammar_plan.v3", "plan_sha256": "reference",
            "dataset": "natural_request", "seed": SEED, "weight_mode": "fitted",
            "source_evidence": "source_text", "search_completions": 4,
            "search_nodes": 256, "max_steps": 8, "training_plan_sha256": "training",
            "checkpoint_receipt_sha256": "checkpoint", "implementation": {"decoder": "fixed"},
            "sources": [identity(example) for example in references]}
    controls = {**base, "schema": "aura.semantic_native_grammar_plan.v11",
                "plan_sha256": "controls", "dataset": "relation_transfer_controls",
                "sources": [identity(case.controlled) for case in cases],
                "implementation": {**base["implementation"],
                    "tools/semantic_native_relation_transfer.py": "cases",
                    "tools/semantic_native_graph_interventions.py": "graph",
                    "tools/semantic_native_paraphrase_interventions.py": "paraphrase",
                    "tools/semantic_native_operation_interventions.py": "operation"}}

    def verification(plan):
        return {"artifacts_verified": True, "current_implementation_drift": [],
                **{key: plan[key] for key in ("plan_sha256", "weight_mode",
                   "training_plan_sha256", "checkpoint_receipt_sha256")}}

    def row(example):
        return {"source_sha256": identity(example), "decode_status": "completed",
                "program_equivalent": True, "answer_correct": True,
                "bound_forced_completion": False, "search": {
                    "observed_program_reach": True, "requested_top_k_proven": True}}

    return (base, controls, verification(base), verification(controls),
            [row(example) for example in references], [row(case.controlled) for case in cases])


def test_twelve_interpretations_establish_the_small_mechanism_without_redecoding_references():
    result = adjudicate_native_relation_transfer(*fixture())
    assert result["mechanism_micro_probe_passed"] is True
    assert result["relations_recovered"] == 9
    assert result["reference_population"] == 3
    assert result["reference_stage_reused_without_redecode"] is True
    assert result["by_kind"] == {kind: {"population": 3, "recovered": 3}
                                for kind in ("paraphrase", "role", "dependency")}
    assert result["general_transfer_proven"] is False


def test_relation_controls_keep_the_residual_calibration_identical_to_reference():
    args = list(deepcopy(fixture()))
    calibration = {"selected_scale": 0.125, "report_receipt_sha256": "calibration"}
    for plan in args[:2]:
        plan["schema"] = "aura.semantic_native_grammar_plan.v13"
        plan["weight_mode"] = "residual"
        plan["residual_calibration"] = calibration
    for verification in args[2:4]:
        verification["weight_mode"] = "residual"
    assert adjudicate_native_relation_transfer(*args)["mechanism_micro_probe_passed"] is True
    args[1]["residual_calibration"] = {**calibration, "report_receipt_sha256": "other"}
    with pytest.raises(ValueError, match="reference protocol"):
        adjudicate_native_relation_transfer(*args)


@pytest.mark.parametrize("defect", ["weights", "budget", "code", "proof", "sources", "unknown"])
def test_changed_candidate_or_incomplete_evidence_cannot_inherit_reference_stage(defect):
    args = list(deepcopy(fixture()))
    if defect == "weights":
        args[1]["checkpoint_receipt_sha256"] = "other"
    elif defect == "budget":
        args[1]["search_nodes"] = 512
    elif defect == "code":
        args[1]["implementation"]["decoder"] = "other"
    elif defect == "proof":
        args[3]["artifacts_verified"] = False
    elif defect == "sources":
        args[5].pop()
    else:
        args[5][0]["answer_correct"] = None
    with pytest.raises(ValueError):
        adjudicate_native_relation_transfer(*args)


@pytest.mark.parametrize("ordinal", range(9))
def test_every_failed_control_stays_at_its_own_mechanism_stage(ordinal):
    args = list(fixture())
    args[5][ordinal]["program_equivalent"] = False
    result = adjudicate_native_relation_transfer(*args)
    assert result["mechanism_micro_probe_passed"] is False
    assert result["relations_recovered"] == 8
    assert result["source_outcomes"][ordinal]["expected_relation_recovered"] is False
