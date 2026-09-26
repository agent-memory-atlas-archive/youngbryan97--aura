"""A durable grammar receipt is not an authority over its own answer."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from core.learning.procedure_induction import Instruction, Program
from tools.verify_semantic_native_grammar import (
    replay_greedy_decisions,
    source_separation_summary,
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

    example = SimpleNamespace(inputs=(8, 3))
    generated = decode_native_grammar(("integer", "integer"),
                                      lambda choices: tuple(1.0 if choice.value in ("add", "input:0", "finish")
                                                            else 0.0 for choice in choices),
                                      register_encoding=REGISTER_ENCODING)
    row = {"decision_trace": json.loads(json.dumps(generated.trace)),
           "program": generated.program.to_dict(),
           "decode_status": "completed", "bound_forced_completion": False}
    plan = {"max_steps": 8, "register_encoding": REGISTER_ENCODING}
    replay_greedy_decisions(row, example=example, plan=plan)
    forged = {**row, "decision_trace": [dict(step) for step in row["decision_trace"]]}
    forged["decision_trace"][0]["chosen"] = "sub"
    with pytest.raises(ValueError, match="winning score"):
        replay_greedy_decisions(forged, example=example, plan=plan)


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
