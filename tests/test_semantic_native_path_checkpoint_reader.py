"""Selection replays every source path, not a scalar or a relabeled receipt."""

import hashlib

import pytest

from core.learning.semantic_native_path_calibration import native_path_totals
from core.learning.semantic_native_path_objective import GRAMMAR_PATH_CONTRACT, path_choice_contract
from core.learning.semantic_native_path_selection import (
    JOINT_GRAPH_SELECTION_CONTRACT,
    PATH_SELECTION_CONTRACT,
)
from tests.test_evaluate_semantic_native_checkpoint import write
from tools.evaluate_semantic_native_checkpoint import digest, selected_checkpoint, verified_document


def campaign(root, *, regression=True, typed=False, graph=False):
    plan = {"schema": "aura.semantic_native_fit_plan.v5", "steps": 2, "save_every": 2,
            "objective": "grammar_choices", "loss_scope": "semantic_decisions",
            "grammar_choice_contract": path_choice_contract(),
            "grammar_path_objective_contract": GRAMMAR_PATH_CONTRACT,
            "path_checkpoint_selection_contract": PATH_SELECTION_CONTRACT,
            "selection": "baseline_preserving_complete_source_calibration_paths",
            "unfitted_checkpoint_eligible": True,
            "held_labels_used_for_fit_or_selection": False,
            "serving_authority": False, "qualification_evidence": False,
            "fit_ids": ["fit"], "captured_fit_ids": ["fit"],
            "calibration_ids": ["cal-a", "cal-b"], "held_ids": ["held"]}
    if typed:
        from tests.test_semantic_native_source_control import typed_plan

        plan.update(typed_plan())
    if graph:
        from core.learning.semantic_native_path_objective import JOINT_GRAPH_CONTRAST_CONTRACT
        from tools.semantic_native_execution import execution_contract

        plan.update(schema="aura.semantic_native_fit_plan.v7",
                    joint_graph_contrast_limit=2,
                    graph_contrast_contract=JOINT_GRAPH_CONTRAST_CONTRACT,
                    path_checkpoint_selection_contract=JOINT_GRAPH_SELECTION_CONTRACT,
                    selection="baseline_preserving_joint_source_calibration",
                    prefix_storage_contract={"mode": "source_shards"},
                    execution_contract=execution_contract(precision="float32", prefix_strategy="trie"))
    write(root / "plan.json", plan, "plan_sha256")
    alternatives = [{"source": source, "decision_index": 0, "choice_index": index,
                     "correct_index": 0, "kind": "termination", "choice": choice}
                    for source in ["fit", "cal-a", "cal-b"]
                    for index, choice in enumerate(["finish", "continue"])]
    supervision = {"plan_sha256": digest(plan), "rows": alternatives}
    if graph:
        supervision["graph_rows"] = [
            {"source": source, "choice_index": index,
             "program_sha256": f"{source}-{index}", "positive": index == 0}
            for source in ["fit", "cal-a", "cal-b"] for index in range(2)]
    write(root / "supervision.json", supervision)
    for step in [0, 2]:
        rows = [{"source": source, "decisions": [{"kind": "termination", "correct_index": 0,
                 "choices": ["finish", "continue"], "scores": scores}]}
                for source, scores in zip(["cal-a", "cal-b"],
                    ([[1., 0.], [0., 1.]] if step == 0 else
                     [[0., 1.] if regression else [1., 0.], [1., 0.]]), strict=True)]
        if graph:
            for row in rows:
                source = row["source"]
                row["whole_graph"] = {"program_sha256s": [f"{source}-0", f"{source}-1"],
                                      "positive_index": 0, "scores": [1., 0.]}
        measured = {"schema": "aura.native_checkpoint_path_calibration.v1",
                    "plan_sha256": digest(plan), "step": step, "rows": rows,
                    "totals": native_path_totals(rows, ["cal-a", "cal-b"])}
        write(root / f"calibration-paths-{step}.json", measured)
        weights = f"weights-{step}".encode()
        (root / f"checkpoint-{step}.safetensors").write_bytes(weights)
        write(root / f"checkpoint-{step}.json", {"plan_sha256": digest(plan), "step": step,
            "calibration_loss": 1. if step == 0 else .01,
            "calibration_path_receipt_sha256": digest(measured),
            "weights_sha256": hashlib.sha256(weights).hexdigest()})
    (root / "rows").mkdir()
    (root / "rows" / "held.json").write_text("held outcomes cannot enter selection")


@pytest.mark.parametrize("regression", [False, True])
@pytest.mark.parametrize("typed", [False, True])
def test_reader_reconstructs_baseline_floor_before_selecting_lower_loss(tmp_path, regression, typed):
    campaign(tmp_path, regression=regression, typed=typed)
    assert selected_checkpoint(tmp_path)[1]["step"] == (0 if regression else 2)


@pytest.mark.parametrize("typed", [False, True])
def test_joint_graph_reader_keeps_typed_or_untyped_v7_and_rejects_rebound_graph(
        tmp_path, monkeypatch, typed):
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    campaign(tmp_path, regression=False, typed=typed, graph=True)
    assert selected_checkpoint(tmp_path)[1]["step"] == 2
    measured_path = tmp_path / "calibration-paths-2.json"
    measured = verified_document(measured_path)
    measured.pop("receipt_sha256")
    measured["rows"][0]["whole_graph"]["program_sha256s"].reverse()
    write(measured_path, measured)
    checkpoint_path = tmp_path / "checkpoint-2.json"
    checkpoint = verified_document(checkpoint_path)
    checkpoint.pop("receipt_sha256")
    checkpoint["calibration_path_receipt_sha256"] = digest(measured)
    write(checkpoint_path, checkpoint)
    with pytest.raises(ValueError, match="whole-graph calibration"):
        selected_checkpoint(tmp_path)


@pytest.mark.parametrize("defect", ["source", "choices", "correct", "kind", "scores", "totals",
                                    "plan", "step", "receipt", "extra", "missing", "supervision"])
def test_reader_refuses_incomplete_or_rebound_path_evidence(tmp_path, defect):
    campaign(tmp_path, regression=False)
    path = tmp_path / "calibration-paths-2.json"
    measured = verified_document(path)
    measured.pop("receipt_sha256")
    decision = measured["rows"][0]["decisions"][0]
    if defect == "source":
        measured["rows"][0]["source"] = "held"
    elif defect == "choices":
        decision["choices"].reverse()
    elif defect == "correct":
        decision["correct_index"] = 1
        measured["totals"] = native_path_totals(measured["rows"], ["cal-a", "cal-b"])
    elif defect == "kind":
        decision["kind"] = "reference"
    elif defect == "scores":
        decision["scores"] = [1., 0., 2.]
        measured["totals"] = native_path_totals(measured["rows"], ["cal-a", "cal-b"])
    elif defect == "totals":
        measured["totals"]["exact_teacher_paths"] = 0
    elif defect == "plan":
        measured["plan_sha256"] = "other"
    elif defect == "step":
        measured["step"] = 0
    elif defect == "extra":
        write(tmp_path / "calibration-paths-6.json", measured)
    elif defect == "missing":
        path.unlink()
    elif defect == "supervision":
        supervision_path = tmp_path / "supervision.json"
        supervision = verified_document(supervision_path)
        supervision.pop("receipt_sha256")
        supervision["rows"] = supervision["rows"][:-1]
        write(supervision_path, supervision)
    if defect != "missing":
        write(path, measured)
    if defect != "receipt":
        checkpoint = verified_document(tmp_path / "checkpoint-2.json")
        checkpoint.pop("receipt_sha256")
        checkpoint["calibration_path_receipt_sha256"] = digest(measured)
        write(tmp_path / "checkpoint-2.json", checkpoint)
    else:
        measured["step"] = 99
        write(path, measured)
    with pytest.raises((ValueError, FileNotFoundError)):
        selected_checkpoint(tmp_path)
