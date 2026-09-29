"""A durable grammar receipt is not an authority over its own answer."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from tools.verify_semantic_native_grammar import (
    replay_greedy_decisions,
    replay_search_decisions,
    source_separation_summary,
    verified_dataset,
    verified_examples,
    verified_input_grounding,
    verified_pair_totals,
    verified_public_inputs,
    verified_source_window,
    verified_weight_mode,
    verify_grammar_row,
    verify_source_separation,
)


def test_grammar_row_reexecutes_and_rejects_forged_success():
    target = Program(2, (Instruction("sub", (0, 1)),))
    rival = Program(2, (Instruction("sub", (1, 0)),))
    example = SimpleNamespace(inputs=(8, 3), program=target, construction_id="novel")
    row = {"source_sha256": "source", "plan_sha256": "plan", "construction": "novel",
           "target_available_to_scorer": False, "program": target.to_dict(),
           "decode_status": "completed", "program_equivalent": True, "answer_correct": True,
           "bound_forced_completion": False, "depth_bound_reached": False}
    assert verify_grammar_row(row, example=example, identity="source", plan_sha256="plan") == (True, True)
    with pytest.raises(ValueError, match="independent execution"):
        verify_grammar_row({**row, "program": rival.to_dict()}, example=example,
                           identity="source", plan_sha256="plan")
    with pytest.raises(ValueError, match="target-blind"):
        verify_grammar_row({**row, "target_available_to_scorer": True}, example=example,
                           identity="source", plan_sha256="plan")
    with pytest.raises(ValueError, match="depth-bound"):
        verify_grammar_row({**row, "depth_bound_reached": True}, example=example,
                           identity="source", plan_sha256="plan")


def test_grammar_row_keeps_incomplete_programs_out_of_success_counts():
    program = Program(2, (Instruction("add", (0, 1)),))
    example = SimpleNamespace(inputs=(8, 3), program=program, construction_id="novel")
    row = {"source_sha256": "source", "plan_sha256": "plan", "construction": "novel",
           "target_available_to_scorer": False, "program": program.to_dict(),
           "decode_status": "disconnected_at_depth_bound", "program_equivalent": False,
           "answer_correct": False, "bound_forced_completion": False,
           "depth_bound_reached": True}
    assert verify_grammar_row(row, example=example, identity="source", plan_sha256="plan") == (False, False)
    with pytest.raises(ValueError, match="independent execution"):
        verify_grammar_row({**row, "answer_correct": True}, example=example,
                           identity="source", plan_sha256="plan")


def test_grammar_decision_replay_rejects_forged_winner():
    from core.learning.semantic_native_grammar import decode_native_grammar
    from core.learning.semantic_native_relative_program import REGISTER_ENCODING

    example = SimpleNamespace(source_text="Use 8 and 3.", inputs=(8, 3))
    generated = decode_native_grammar(("integer", "integer"),
                                      lambda choices: tuple(1.0 if choice.value in ("add", "input:0", "finish")
                                                            else 0.0 for choice in choices),
                                      register_encoding=REGISTER_ENCODING)
    row = {"decision_trace": json.loads(json.dumps(generated.trace)),
           "program": generated.program.to_dict(),
           "decode_status": "completed", "bound_forced_completion": False}
    plan = {"schema": "aura.semantic_native_grammar_plan.v5",
            "max_steps": 8, "register_encoding": REGISTER_ENCODING}
    replay_greedy_decisions(row, example=example, plan=plan)
    forged = {**row, "decision_trace": [dict(step) for step in row["decision_trace"]]}
    forged["decision_trace"][0]["chosen"] = "sub"
    with pytest.raises(ValueError, match="winning score"):
        replay_greedy_decisions(forged, example=example, plan=plan)


def test_search_replay_rebuilds_discarded_branches_and_rejects_forged_selection():
    from core.learning.semantic_native_relative_program import REGISTER_ENCODING
    from core.learning.semantic_native_search import search_native_grammar
    from core.learning.semantic_program_floor import semantic_programs_structurally_equivalent

    transcript = []
    receipts = []

    def score(choices):
        values = [choice.value for choice in choices]
        scores = [0.0] * len(choices)
        transcript.append({"choices": values, "scores": scores})
        receipts.append([{"choice": value} for value in values])
        return tuple(scores)

    plan = {"max_steps": 3, "search_nodes": 64, "search_completions": 4,
            "register_encoding": REGISTER_ENCODING,
            "search_score_mode": "normalized_choices"}
    searched = search_native_grammar(("integer", "integer"), score,
        max_steps=plan["max_steps"], max_nodes=plan["search_nodes"],
        completions=plan["search_completions"],
        register_encoding=plan["register_encoding"],
        score_mode=plan["search_score_mode"])
    assert len(searched.candidates) == 4
    chosen = 2
    selected = searched.candidates[chosen].result
    example = SimpleNamespace(program=searched.candidates[0].result.program)
    search = {"expanded_nodes": searched.expanded_nodes,
              "scored_decisions": searched.scored_decisions,
              "scored_alternatives": searched.scored_alternatives,
              "disconnected_leaves": searched.disconnected_leaves,
              "pruned_prefixes": searched.pruned_prefixes,
              "frontier_nodes": searched.frontier_nodes,
              "frontier_log_probability_bound": searched.frontier_log_probability_bound,
              "halt_reason": searched.halt_reason,
              "requested_top_k_proven": searched.requested_top_k_proven,
              "score_transcript": transcript,
              "graph_score_input_receipts": [
                  {"program": candidate.result.program.to_dict()}
                  for candidate in searched.candidates],
              "selected_index": chosen,
              "complete_graph_scores": [0.0, 1.0, 2.0, -1.0],
              "proposals": [{"program": candidate.result.program.to_dict(),
                  "log_probability": candidate.log_probability,
                  "decision_trace": candidate.result.trace,
                  "bound_forced_completion": candidate.result.bound_forced_completion}
                  for candidate in searched.candidates],
              "observed_program_reach": any(semantic_programs_structurally_equivalent(
                  candidate.result.program, example.program) for candidate in searched.candidates)}
    row = json.loads(json.dumps({"search": search, "score_input_receipts": receipts,
        "decode_status": "completed", "program": selected.program.to_dict(),
        "decision_trace": selected.trace,
        "bound_forced_completion": selected.bound_forced_completion}))

    def expected_receipts(choices):
        return [{"choice": choice.value} for choice in choices]

    def verify(candidate):
        replay_search_decisions(candidate, example=example, plan=plan,
            input_types=("integer", "integer"),
            input_receipts_for_choices=expected_receipts,
            input_receipt_for_program=lambda program: {"program": program.to_dict()})

    verify(row)
    for path, value in (
        (("search", "score_transcript", 0, "choices", 0), "forged"),
        (("search", "score_transcript", 0, "scores", 0), 7.0),
        (("score_input_receipts", 0, 0, "choice"), "forged"),
        (("search", "selected_index"), 0),
        (("search", "graph_score_input_receipts", 0, "program"),
         selected.program.to_dict()),
        (("search", "proposals", 0, "program"), selected.program.to_dict()),
        (("program",), searched.candidates[0].result.program.to_dict()),
    ):
        forged = json.loads(json.dumps(row))
        target = forged
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match="native search"):
            verify(forged)
    forged = json.loads(json.dumps(row))
    forged["search"]["score_transcript"].append(forged["search"]["score_transcript"][0])
    forged["score_input_receipts"].append(forged["score_input_receipts"][0])
    with pytest.raises(ValueError, match="unvisited decisions"):
        verify(forged)


@pytest.mark.parametrize("mode", ["source_text", "source_token_erasure", "source_pair_swap"])
def test_grammar_replay_binds_each_score_to_the_scored_input(mode):
    from core.learning.semantic_native_grammar import decode_native_grammar
    from core.learning.semantic_native_program import native_text_decision_sequence
    from core.learning.semantic_native_relative_program import REGISTER_ENCODING
    from core.learning.semantic_native_source_control import (
        apply_native_source_evidence,
        native_score_input_receipt,
    )
    from tests.test_semantic_native_program import Tokenizer

    example = SimpleNamespace(source_text="Add 8 and 3.", inputs=(8, 3))
    scored_source = ("Subtract 8 from 3." if mode == "source_pair_swap"
                     else example.source_text)
    tokenizer = Tokenizer()
    input_receipts = []

    def score(choices):
        receipts = []
        for choice in choices:
            sequence = native_text_decision_sequence(
                scored_source, choice.text, (choice.span,), tokenizer,
                max_tokens=1024)
            sequence, control = apply_native_source_evidence(
                sequence, scored_source, tokenizer,
                mode="source_text" if mode == "source_pair_swap" else mode)
            receipts.append(native_score_input_receipt(sequence, control))
        input_receipts.append(receipts)
        return tuple(1.0 if choice.value in ("add", "input:0", "finish") else 0.0
                     for choice in choices)

    generated = decode_native_grammar(("integer", "integer"), score,
                                      register_encoding=REGISTER_ENCODING)
    row = {"decision_trace": json.loads(json.dumps(generated.trace)),
           "score_input_receipts": json.loads(json.dumps(input_receipts)),
           "program": generated.program.to_dict(), "decode_status": "completed",
           "bound_forced_completion": False}
    plan = {"schema": "aura.semantic_native_grammar_plan.v5",
            "max_steps": 8, "register_encoding": REGISTER_ENCODING,
            "source_evidence": mode}
    replay_greedy_decisions(row, example=example, plan=plan, tokenizer=tokenizer,
                            max_sequence_tokens=1024, scored_source=scored_source)
    forged = json.loads(json.dumps(row))
    forged["score_input_receipts"][0][0]["sequence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scored-input tokens"):
        replay_greedy_decisions(forged, example=example, plan=plan,
                                tokenizer=tokenizer, max_sequence_tokens=1024,
                                scored_source=scored_source)
    with pytest.raises(ValueError, match="receipts are missing"):
        replay_greedy_decisions(row, example=example, plan=plan)


def test_source_separation_requires_text_and_construction_disjointness():
    source = [SimpleNamespace(source_text="old request", construction_id="two-step",
                              topology_id="shared relation")]
    target = [SimpleNamespace(source_text="new request", construction_id="three-step",
                              topology_id="shared relation")]
    assert source_separation_summary(source, target) == {
        "source_examples": 1, "source_constructions": 1, "target_constructions": 1,
        "shared_topologies": ["shared relation"],
        "source_text_overlap": 0, "construction_overlap": 0,
    }
    with pytest.raises(ValueError, match="source text overlaps"):
        source_separation_summary(source, [SimpleNamespace(source_text="old request",
                                                           construction_id="three-step",
                                                           topology_id="shared relation")])
    with pytest.raises(ValueError, match="construction overlaps"):
        source_separation_summary(source, [SimpleNamespace(source_text="new request",
                                                           construction_id="two-step",
                                                           topology_id="shared relation")])


def test_source_separation_binds_report_and_rebuilt_manifest(tmp_path, monkeypatch):
    from core.learning.semantic_program_campaign import _sha

    source = SimpleNamespace(source_text="old request", construction_id="two-step",
                             topology_id="old topology")
    target = SimpleNamespace(source_text="new request", construction_id="three-step",
                             topology_id="new topology")
    monkeypatch.setattr(
        "core.learning.semantic_program_feature_materialization.rebuild_semantic_feature_selection",
        lambda manifest: (None, (source,)),
    )
    bundles = [tmp_path / f"source_{index}" for index in range(3)]
    manifests = {f"source_{index}": f"bound-manifest-{index}" for index in range(3)}
    for bundle in bundles:
        bundle.mkdir()
        (bundle / "manifest.json").write_text(json.dumps({
            "manifest_sha256": manifests[bundle.name]}))
    report_body = {"fit_complete": True, "representation_compatibility": {
        "source_feature_manifest_sha256s": manifests}}
    report = {**report_body, "report_sha256": _sha(report_body)}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    training = {"source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest()}
    arguments = [f"{bundle.name}={bundle}" for bundle in bundles]
    result = verify_source_separation(training, report_path, arguments, [target])
    assert result["source_examples"] == 3
    assert result["construction_overlap"] == 0

    (bundles[0] / "manifest.json").write_text(json.dumps({"manifest_sha256": "changed"}))
    with pytest.raises(ValueError, match="manifest differs"):
        verify_source_separation(training, report_path, arguments, [target])
    (bundles[0] / "manifest.json").write_text(json.dumps({
        "manifest_sha256": manifests[bundles[0].name]}))
    report_path.write_text(json.dumps({**report, "fit_complete": False}))
    with pytest.raises(ValueError, match="source report differs"):
        verify_source_separation(training, report_path, arguments, [target])


def test_weight_mode_verification_keeps_base_and_fitted_arms_distinct():
    old_plan = {"schema": "aura.semantic_native_grammar_plan.v1"}
    old_report = {"schema": "aura.semantic_native_grammar.v1"}
    assert verified_weight_mode(old_plan, old_report) == "fitted"
    with pytest.raises(ValueError, match="historical"):
        verified_weight_mode({**old_plan, "weight_mode": "base"}, old_report)
    plan = {"schema": "aura.semantic_native_grammar_plan.v2", "weight_mode": "base"}
    report = {"schema": "aura.semantic_native_grammar.v2", "weight_mode": "base"}
    assert verified_weight_mode(plan, report) == "base"
    with pytest.raises(ValueError, match="weight mode differs"):
        verified_weight_mode(plan, {**report, "weight_mode": "fitted"})
    with pytest.raises(ValueError, match="schema versions differ"):
        verified_weight_mode(plan, old_report)
    next_plan = {"schema": "aura.semantic_native_grammar_plan.v3", "weight_mode": "fitted"}
    next_report = {"schema": "aura.semantic_native_grammar.v3", "weight_mode": "fitted"}
    assert verified_weight_mode(next_plan, next_report) == "fitted"
    next_plan["schema"] = "aura.semantic_native_grammar_plan.v4"
    next_report["schema"] = "aura.semantic_native_grammar.v4"
    assert verified_weight_mode(next_plan, next_report) == "fitted"
    residual_plan = {"schema": "aura.semantic_native_grammar_plan.v9", "weight_mode": "residual"}
    residual_report = {"schema": "aura.semantic_native_grammar.v9", "weight_mode": "residual"}
    assert verified_weight_mode(residual_plan, residual_report) == "residual"
    with pytest.raises(ValueError, match="weight mode differs"):
        verified_weight_mode({**residual_plan, "weight_mode": "fitted"}, residual_report)
    with pytest.raises(ValueError, match="weight mode differs"):
        verified_weight_mode({**plan, "weight_mode": "residual"},
                             {**report, "weight_mode": "residual"})


def test_full_prefix_joint_residual_has_one_receipt_bound_mode_across_cohorts():
    plan = {"schema": "aura.semantic_native_grammar_plan.v13", "weight_mode": "residual",
            "dataset": "natural_request", "seed": 31}
    report = {"schema": "aura.semantic_native_grammar.v13", "weight_mode": "residual",
              "dataset": "natural_request", "seed": 31}
    assert verified_weight_mode(plan, report) == "residual"
    assert verified_dataset(plan, report) == ("natural_request", 31)
    assert verified_source_window(plan, report) is None
    with pytest.raises(ValueError, match="weight mode differs"):
        verified_weight_mode({**plan, "weight_mode": "fitted"}, report)
    with pytest.raises(ValueError, match="source window"):
        verified_dataset({**plan, "source_window": {}}, report)

    controls = {**plan, "dataset": "relation_transfer_controls"}
    assert verified_dataset(controls, {**report, "dataset": "relation_transfer_controls"}) == (
        "relation_transfer_controls", 31)
    basis = {"split": "validation", "source_report_sha256": "a" * 64}
    retained = {**plan, "dataset": "retained_validation", "seed": 0,
                "source_cohort_basis": basis, "sources": ["s"]}
    retained_report = {**report, "dataset": "retained_validation", "seed": 0,
                       "source_cohort_basis": basis}
    assert verified_dataset(retained, retained_report) == ("retained_validation", 0)
    window = {"offset": 499, "count": 1, "population": 500,
              "ordered_sources_sha256": "b" * 64}
    retained["source_window"] = window
    retained_report["source_window"] = window
    assert verified_source_window(retained, retained_report) == window
    with pytest.raises(ValueError, match="source window"):
        verified_source_window(retained, {**retained_report, "source_window": {**window, "count": 2}})


def test_public_values_recovered_from_source_not_assumed_from_annotation():
    example = SimpleNamespace(source_text="Use [7, -2] and 3.", inputs=((7, -2), 3))
    assert verified_public_inputs(example) == ((7, -2), 3)
    with pytest.raises(ValueError, match="differ from annotations"):
        verified_public_inputs(SimpleNamespace(source_text=example.source_text,
                                               inputs=((7, -2), 4)))


def test_grounding_claim_cannot_change_without_a_protocol_version():
    plan = {"input_grounding": "declared_public_inputs"}
    report = {"input_grounding": "declared_public_inputs"}
    assert verified_input_grounding(plan, report) == "declared_public_inputs"
    with pytest.raises(ValueError, match="input grounding differs"):
        verified_input_grounding(plan, {"input_grounding": "semantic_public_character_inputs.v1"})
    with pytest.raises(ValueError, match="input grounding differs"):
        verified_input_grounding({"input_grounding": "semantic_public_character_inputs.v1"}, report)
    new_plan = {"schema": "aura.semantic_native_grammar_plan.v3",
                "input_grounding": "semantic_public_character_inputs.v1"}
    new_report = {"input_grounding": "semantic_public_character_inputs.v1"}
    assert verified_input_grounding(new_plan, new_report) == new_report["input_grounding"]
    new_plan["schema"] = "aura.semantic_native_grammar_plan.v4"
    assert verified_input_grounding(new_plan, new_report) == new_report["input_grounding"]
    with pytest.raises(ValueError, match="input grounding differs"):
        verified_input_grounding(new_plan, report)


def test_intervention_dataset_reconstruction_is_independent_of_evaluator_order():
    plan = {"schema": "aura.semantic_native_grammar_plan.v3", "dataset": "operation_intervention",
            "seed": 2718283, "sources": ["source"] * 6}
    report = {"dataset": "operation_intervention", "seed": 2718283}
    assert verified_dataset(plan, report) == ("operation_intervention", 2718283)
    examples = verified_examples(plan, dataset="operation_intervention", seed=2718283)
    assert len(examples) == 6
    assert {example.topology_id for example in examples} == {
        "scalar_linear_three", "lookup_linear_three", "count_linear_three",
    }
    assert all(examples[index].source_text != examples[index + 1].source_text
               for index in range(0, 6, 2))
    with pytest.raises(ValueError, match="seed differs"):
        verified_dataset(plan, {**report, "seed": 2718285})
    with pytest.raises(ValueError, match="split a source pair"):
        verified_examples({**plan, "sources": ["source"] * 5},
                          dataset="operation_intervention", seed=2718283)
    old_plan = {"schema": "aura.semantic_native_grammar_plan.v2"}
    assert verified_dataset(old_plan, {}) == ("natural_request", 3141592)


@pytest.mark.parametrize("dataset", ["definition_intervention", "equation_intervention"])
def test_v4_paraphrase_dataset_rebuilds_without_admitting_v3_labels(dataset):
    plan = {"schema": "aura.semantic_native_grammar_plan.v4", "dataset": dataset,
            "seed": 2718283, "sources": ["source"] * 6}
    report = {"dataset": dataset, "seed": 2718283}
    assert verified_dataset(plan, report) == (dataset, 2718283)
    examples = verified_examples(plan, dataset=dataset, seed=2718283)
    assert len(examples) == 6
    assert {item.topology_id for item in examples} == {
        "scalar_linear_three", "lookup_linear_three", "count_linear_three",
    }
    assert all(verified_public_inputs(item) == item.inputs for item in examples)
    with pytest.raises(ValueError, match="dataset or seed"):
        verified_dataset({**plan, "schema": "aura.semantic_native_grammar_plan.v3"}, report)


def test_independent_intervention_pair_metrics_reject_wrong_registers():
    before = {"program_equivalent": True, "answer_correct": True,
              "decode_status": "completed", "program": {"instructions": [
                  ["add", [0, 1]], ["sub", [4, 2]]]}}
    after = {**before, "program": {"instructions": [
        ["add", [0, 1]], ["add", [4, 2]]]}}
    assert verified_pair_totals((before, after), dataset="operation_intervention") == {
        "pair_count": 1, "pair_exact": 1, "source_responsive": 1,
    }
    wrong = {**after, "program": {"instructions": [
        ["add", [0, 1]], ["add", [4, 3]]]}}
    assert verified_pair_totals((before, wrong), dataset="operation_intervention")["source_responsive"] == 0


@pytest.mark.parametrize("version", ["v6", "v8", "v9"])
def test_retained_development_protocol_cannot_be_relabelled_fresh(version):
    basis = {"split": "test", "exposure": "previously_exposed_development"}
    plan = {"schema": f"aura.semantic_native_grammar_plan.{version}", "dataset": "retained_test",
            "seed": 0, "source_cohort_basis": basis}
    report = {"dataset": "retained_test", "seed": 0, "source_cohort_basis": basis}
    assert verified_dataset(plan, report) == ("retained_test", 0)
    with pytest.raises(ValueError, match="source basis differs"):
        verified_dataset(plan, {**report, "source_cohort_basis": {**basis, "exposure": "fresh"}})
    with pytest.raises(ValueError, match="source basis differs"):
        verified_dataset({**plan, "seed": 1}, {**report, "seed": 1})
    with pytest.raises(ValueError, match="historical"):
        verified_dataset({**plan, "schema": "aura.semantic_native_grammar_plan.v5"}, report)
    with pytest.raises(ValueError, match="dataset or seed"):
        verified_dataset({**plan, "dataset": "natural_request"}, {**report, "dataset": "natural_request"})


def test_retained_examples_rebuild_the_pinned_basis_before_grading(monkeypatch):
    basis = {"split": "test", "source_report_path": "report.json",
             "source_manifest_paths": {"a": "bundle"}, "available_population": 500,
             "exposure": "previously_exposed_development"}
    plan = {"sources": ["a", "b"], "source_cohort_basis": basis}
    calls = []
    def load(path, bundles, *, split, count):
        calls.append((path, bundles, split, count))
        return ("first", "second"), dict(basis)
    monkeypatch.setattr("tools.semantic_native_retained_sources.load_retained_native_sources", load)
    assert verified_examples(plan, dataset="retained_test", seed=0) == ("first", "second")
    assert calls == [("report.json", ["a=bundle"], "test", 2)]
    with pytest.raises(ValueError, match="split differs"):
        verified_examples(plan, dataset="retained_validation", seed=0)
    monkeypatch.setattr("tools.semantic_native_retained_sources.load_retained_native_sources",
        lambda *args, **kwargs: ((), {**basis, "available_population": 2}))
    with pytest.raises(ValueError, match="reconstruction differs"):
        verified_examples(plan, dataset="retained_test", seed=0)
