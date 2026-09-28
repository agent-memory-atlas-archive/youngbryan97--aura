"""Selection replays every source path, not a scalar or a relabeled receipt."""

import hashlib

import pytest

from core.learning.semantic_native_path_calibration import native_path_totals
from core.learning.semantic_native_path_objective import GRAMMAR_PATH_CONTRACT, path_choice_contract
from core.learning.semantic_native_path_selection import PATH_SELECTION_CONTRACT
from tests.test_evaluate_semantic_native_checkpoint import write
from tools.evaluate_semantic_native_checkpoint import digest, selected_checkpoint, verified_document


def campaign(root, *, regression=True, typed=False):
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
    write(root / "plan.json", plan, "plan_sha256")
    alternatives = [{"source": source, "decision_index": 0, "choice_index": index,
                     "correct_index": 0, "kind": "termination", "choice": choice}
                    for source in ["fit", "cal-a", "cal-b"]
                    for index, choice in enumerate(["finish", "continue"])]
    write(root / "supervision.json", {"plan_sha256": digest(plan), "rows": alternatives})
    for step in [0, 2]:
        rows = [{"source": source, "decisions": [{"kind": "termination", "correct_index": 0,
                 "choices": ["finish", "continue"], "scores": scores}]}
                for source, scores in zip(["cal-a", "cal-b"],
                    ([[1., 0.], [0., 1.]] if step == 0 else
                     [[0., 1.] if regression else [1., 0.], [1., 0.]]), strict=True)]
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
