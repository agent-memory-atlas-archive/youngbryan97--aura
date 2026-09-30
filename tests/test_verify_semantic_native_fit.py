"""The independent native verifier rejects outcome-count and witness forgery."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.verify_semantic_native_fit import (
    persisted_source_pairs,
    regrade_bank,
    verify_native_totals,
    verify_native_source_partition,
    verify_source_control_supervision,
)


@pytest.mark.parametrize("defect", [None, "held_fit", "missing_fit", "duplicate",
    "bank_overlap", "foreign_calibration", "foreign_held", "foreign_donor", "uncaptured"])
def test_native_partition_is_bound_to_the_independent_bank(defect):
    bank = {"fit_ids": ["a", "b"], "calibration_ids": ["c", "d"], "held_ids": ["e", "f"]}
    plan = {"fit_ids": ["a", "b"], "calibration_ids": ["c"], "held_ids": ["e"],
        "captured_fit_ids": ["a", "b"], "scheduled_fit_ids": ["a", "a"]}
    if defect is None:
        verify_native_source_partition(plan, bank)
        return
    if defect == "held_fit":
        plan["fit_ids"] = ["a", "e"]
    elif defect == "missing_fit":
        plan["fit_ids"].pop()
    elif defect == "duplicate":
        plan["calibration_ids"].append("c")
    elif defect == "bank_overlap":
        bank["held_ids"].append("a")
    elif defect == "foreign_calibration":
        plan["calibration_ids"] = ["e"]
    elif defect == "foreign_held":
        plan["held_ids"] = ["c"]
    elif defect == "foreign_donor":
        plan["captured_fit_ids"].append("e")
    else:
        plan["captured_fit_ids"] = ["b"]
    with pytest.raises(ValueError, match="partition|independent bank"):
        verify_native_source_partition(plan, bank)


def test_source_pair_witness_compares_persisted_sequence_inputs():
    raw = {"source": [{"partner": "peer", "witness": {
        "inputs": [(4, 7), 2], "outputs": [(4, 7), 2]}}]}
    persisted = persisted_source_pairs(raw)
    assert raw != persisted
    assert persisted == {"source": [{"partner": "peer", "witness": {
        "inputs": [[4, 7], 2], "outputs": [[4, 7], 2]}}]}
    forged = deepcopy(persisted)
    forged["source"][0]["witness"]["outputs"][0][0] = 8
    assert persisted_source_pairs(raw) != forged


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


def source_control_fixture():
    import hashlib
    import json

    from core.learning.procedure_induction import Instruction, Program
    from core.learning.semantic_native_source_control import SOURCE_ERASURE_CONTRACT
    from tools.train_semantic_native_program import native_supervision_sets
    from tests.test_semantic_native_program import Tokenizer

    tokenizer = Tokenizer()
    target = Program(2, (Instruction("sub", (0, 1)),))
    texts = {hashlib.sha256(text.encode()).hexdigest(): text
             for text in ("Subtract 2 from 5.", "Take 3 away from 7.")}
    fit, calibration = tuple(texts)
    items = {key: SimpleNamespace(split="train", public_inputs=(5, 2), ir=SimpleNamespace(
        source_text_sha256=key, source_token_ids=tuple(tokenizer.encode(text)),
        to_program=lambda: target)) for key, text in texts.items()}
    receipts = {}
    sequences, _ = native_supervision_sets(items, texts, tokenizer, tuple(texts),
        source_erasure_ids=(fit,), source_control_receipts=receipts, contrast_limit=4, peers=(target,))
    plan = {"schema": "aura.semantic_native_fit_plan.v2", "source_evidence_control": SOURCE_ERASURE_CONTRACT,
        "fit_ids": [fit], "captured_fit_ids": [fit], "calibration_ids": [calibration],
        "supervision_peer_program_sha256s": [target.sha()], "objective": "contrastive", "contrast_limit": 4,
        "max_sequence_tokens": 1024, "register_encoding": "absolute_v1"}
    supervision = {"source_evidence_control": {**SOURCE_ERASURE_CONTRACT,
        "erased_fit_ids": [fit], "unchanged_calibration_ids": [calibration]},
        "rows": [{"source": key[0], "program_sha256": key[1], "tokens": row.tokens,
                  "continuation_start": row.continuation_start, "semantic_positions": row.semantic_positions,
                  "source_control_receipt": receipts.get(key)} for key, row in sorted(sequences.items())]}
    return plan, json.loads(json.dumps(supervision)), items, tokenizer


def test_independent_source_control_reconstructs_all_tokens_and_scope():
    plan, supervision, items, tokenizer = source_control_fixture()
    verified = verify_source_control_supervision(plan, supervision, items, tokenizer)
    assert verified["source_control_verified"] is True
    assert verified["erased_fit_population"] == verified["intact_calibration_population"] == 1
    assert verified["supervision_sequences_verified"] == len(supervision["rows"])


@pytest.mark.parametrize("defect", ["source_token", "target_token", "receipt", "calibration", "scope", "duplicate", "tokenizer"])
def test_independent_source_control_rejects_erasure_or_scope_forgery(defect):
    plan, supervision, items, tokenizer = source_control_fixture()
    fit_row = next(row for row in supervision["rows"] if row["source"] in plan["captured_fit_ids"])
    cal_row = next(row for row in supervision["rows"] if row["source"] in plan["calibration_ids"])
    if defect == "source_token":
        fit_row["tokens"][fit_row["source_control_receipt"]["erased_source_positions"][0]] = ord("S")
    elif defect == "target_token":
        fit_row["tokens"][-1] += 1
    elif defect == "receipt":
        fit_row["source_control_receipt"] = None
    elif defect == "calibration":
        cal_row["source_control_receipt"] = fit_row["source_control_receipt"]
    elif defect == "scope":
        supervision["source_evidence_control"]["erased_fit_ids"] += plan["calibration_ids"]
    elif defect == "duplicate":
        supervision["rows"].append(supervision["rows"][0])
    else:
        tokenizer = None
    with pytest.raises(ValueError, match="source control"):
        verify_source_control_supervision(plan, supervision, items, tokenizer)


def test_historical_fits_are_not_reinterpreted_as_source_erasure_controls():
    plan = {"schema": "aura.semantic_native_fit_plan.v1"}
    assert verify_source_control_supervision(plan, {"rows": []}, {}, None)["mode"] == "source_text"
    with pytest.raises(ValueError, match="silently erase"):
        verify_source_control_supervision(plan, {"source_evidence_control": {}, "rows": []}, {}, None)
