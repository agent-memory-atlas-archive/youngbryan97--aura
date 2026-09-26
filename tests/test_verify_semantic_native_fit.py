"""The independent native verifier rejects outcome-count and witness forgery."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.verify_semantic_native_fit import regrade_bank, verify_native_totals


def test_totals_keep_unknowns_separate_from_correct_and_regressed():
    rows = [{"source": "a", "incumbent_correct": True, "selected_correct": None,
             "pretrained_correct": False, "bank_reachable": None},
            {"source": "b", "incumbent_correct": False, "selected_correct": True,
             "pretrained_correct": None, "bank_reachable": True}]
    report = {"rows": rows, "population": 2, "incumbent_correct": 1, "native_correct": 1,
        "pretrained_correct": 0, "bank_reachable": 1, "gains": 1, "regressions": 0,
        "learning_gains": 0, "learning_regressions": 0}
    checked = verify_native_totals(report, rows)
    assert checked["unknown_selected"] == checked["unknown_pretrained"] == 1
    for field, value in (("native_correct", 2), ("regressions", 1), ("population", True)):
        with pytest.raises(ValueError, match="totals"):
            verify_native_totals({**report, field: value}, rows)
    with pytest.raises(ValueError, match="repeat"):
        verify_native_totals({**report, "rows": rows + rows}, rows + rows)


def test_semantic_regrading_reexecutes_programs_instead_of_accepting_cached_labels(monkeypatch):
    from core.learning.procedure_induction import Instruction, Program
    from core.learning import semantic_joint_graph_learning as alignment

    target = Program(2, (Instruction("add", (0, 1)),))
    rival = Program(2, (Instruction("sub", (0, 1)),))
    monkeypatch.setattr(alignment, "align_source_input_registers", lambda *args: (target.instructions, ()))
    candidates = [{"program_sha256": program.sha(), "program": program.to_dict()} for program in (target, rival)]
    bank = {"bank": {"input_spans": [{"start": 0, "end": 1}, {"start": 1, "end": 2}],
                     "candidates": candidates},
            "diagnosis": {"target_program_sha256": target.sha(), "comparisons": [
                {"program_sha256": target.sha(), "status": "equivalent"},
                {"program_sha256": rival.sha(), "status": "different"}]}}
    item = SimpleNamespace(public_inputs=(2, 3))
    labels, counts = regrade_bank(bank, item)
    assert labels == {target.sha(): True, rival.sha(): False}
    assert counts == {"equivalent": 1, "different": 1}
    forged = deepcopy(bank)
    forged["diagnosis"]["comparisons"][1]["status"] = "equivalent"
    with pytest.raises(ValueError, match="independent execution"):
        regrade_bank(forged, item)
