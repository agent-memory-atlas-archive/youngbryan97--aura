"""Typed parameter isolation preserves measured paths, never supplies a target."""

from copy import deepcopy
import json

import pytest

from core.learning.semantic_native_factorized_residual import (
    _native_parameter_route_types,
    factorized_paths,
    factorized_residual_admission,
    native_competition_kind,
    validated_kind_scales,
)
from core.learning.semantic_native_grammar import NativeGrammarDecision, decode_native_grammar
from tools.verify_semantic_native_factorized_residual import independent_factor_admission
from tools.verify_semantic_native_grammar import verify_factorized_parameter_receipt


def paths():
    def row(source, scores):
        return {"source": source, "decisions": [{"kind": kind, "choices": choices,
            "correct_index": 0, "scores": values}
            for kind, choices, values in zip(("operation", "reference", "termination"),
                (("add", "sub"), ("input:0", "input:1"), ("finish", "continue")), scores, strict=True)]}
    base = [row("a", ((2., 0.), (2., 0.), (2., 0.))),
            row("b", ((0., 2.), (2., 0.), (2., 0.)))]
    fit = [row("a", ((2., 0.), (0., 2.), (2., 0.))),
           row("b", ((2., 0.), (2., 0.), (2., 0.)))]
    return {0.: base, 1.: fit}, ["a", "b"]


def test_source_admission_keeps_operation_gain_without_binding_regression():
    measured, sources = paths()
    result = factorized_residual_admission(measured, sources)
    assert result == independent_factor_admission(measured, sources)
    assert result["selected_scales"] == {"operation": 1., "reference": 0., "termination": 0.}
    assert result["selected_totals"]["exact_teacher_paths"] == 2
    assert len(result["adjudication"]) == 8
    all_fit = next(row for row in result["adjudication"] if all(row["scales"].values()))
    assert not all_fit["eligible"] and all_fit["lost_baseline_sources"] == ["a"]


@pytest.mark.parametrize("defect", ["missing_source", "duplicate_source", "choices", "positive", "kind", "nan", "path"])
def test_factorization_cannot_hide_source_or_supervision_drift(defect):
    measured, sources = paths()
    row = measured[1.][0]
    if defect == "missing_source":
        measured[1.].pop()
    elif defect == "duplicate_source":
        measured[1.][1]["source"] = "a"
    elif defect == "choices":
        row["decisions"][0]["choices"] = ("sub", "add")
    elif defect == "positive":
        row["decisions"][0]["correct_index"] = 1
    elif defect == "kind":
        row["decisions"][0]["kind"] = "reference"
    elif defect == "nan":
        row["decisions"][0]["scores"] = (float("nan"), 0.)
    else:
        row["decisions"].pop(0)
    with pytest.raises(ValueError):
        factorized_residual_admission(measured, sources)


def test_no_unmeasured_interpolation_or_partial_kind_policy():
    measured, sources = paths()
    with pytest.raises(ValueError, match="unmeasured"):
        factorized_paths(measured, sources, {"operation": .5, "reference": 0., "termination": 1.})
    for scales in ({"operation": 1.}, {"operation": True, "reference": 0., "termination": 1.},
                   {"operation": float("nan"), "reference": 0., "termination": 1.}):
        with pytest.raises(ValueError):
            validated_kind_scales(scales)


def choice(value):
    return NativeGrammarDecision("unused", (0, 1), value)


@pytest.mark.parametrize("values,kind", [(("add", "sub"), "operation"),
    (("finish", "continue"), "termination"), ((0, 1), "reference"),
    (("input:0", "result:0"), "reference"), (("input:0",), "reference")])
def test_kind_is_a_grammar_type_without_source_or_family_evidence(values, kind):
    assert native_competition_kind(tuple(choice(value) for value in values)) == kind


@pytest.mark.parametrize("values", [(True,), (-1,), ("add", "finish"),
                                    ("unknown",), ("input:01",), ("input:0", 1), (0, 0)])
def test_undeclared_or_mixed_atom_types_do_not_route(values):
    with pytest.raises(ValueError):
        native_competition_kind(tuple(choice(value) for value in values))


def test_runtime_grammar_reports_the_same_types_as_the_isolated_scorer():
    seen = []
    references = 0
    def score(choices):
        nonlocal references
        kind = native_competition_kind(choices)
        seen.append(kind)
        value = "sub" if kind == "operation" else "finish" if kind == "termination" else references
        references += kind == "reference"
        return tuple(1. if item.value == value else 0. for item in choices)
    result = decode_native_grammar(("integer", "integer"), score, max_steps=2)
    assert seen == [entry["kind"] for entry in result.trace]
    assert result.program.run((8, 3)) == 5


@pytest.mark.parametrize("defect", [None, "kind", "scale", "sites", "trace"])
def test_parameter_receipts_are_bound_to_the_actual_competition(defect):
    plan = {"factorized_residual": {"selected_scales": {
        "operation": 1., "reference": .08, "termination": 1.}}}
    entry = {"kind": "reference"}
    receipt = {"kind": "reference", "scale": .08, "adapter_sites": 4}
    if defect == "kind":
        receipt["kind"] = "operation"
    elif defect == "scale":
        receipt["scale"] = 1.
    elif defect == "sites":
        receipt["adapter_sites"] = 0
    elif defect == "trace":
        entry["kind"] = "operation"
    choices = (choice("input:0"), choice("input:1"))
    if defect is None:
        verify_factorized_parameter_receipt(plan, entry, choices, receipt)
    else:
        with pytest.raises(ValueError):
            verify_factorized_parameter_receipt(plan, entry, choices, receipt)


def test_selection_does_not_mutate_measured_source_paths():
    measured, sources = paths()
    before = deepcopy(measured)
    factorized_residual_admission(measured, sources)
    assert measured == before


def test_registered_route_invariant_runs_the_real_type_reader():
    assert _native_parameter_route_types() == ("operation", "reference", "termination")


def test_new_decode_schema_cannot_be_used_as_an_unfitted_or_global_residual_arm():
    from tools.verify_semantic_native_grammar import verified_weight_mode

    plan = {"schema": "aura.semantic_native_grammar_plan.v10", "weight_mode": "factorized"}
    report = {"schema": "aura.semantic_native_grammar.v10", "weight_mode": "factorized"}
    assert verified_weight_mode(plan, report) == "factorized"
    for mode in ("base", "fitted", "residual"):
        with pytest.raises(ValueError):
            verified_weight_mode({**plan, "weight_mode": mode}, {**report, "weight_mode": mode})


@pytest.mark.parametrize("defect", [None, "selection", "missing_option", "authority", "sources", "implementation"])
def test_factor_receipt_is_rebuilt_from_bound_measurements(tmp_path, monkeypatch, defect):
    from tools.evaluate_semantic_native_checkpoint import digest
    from tools.factor_semantic_native_residual import factor_residual
    from tools.verify_semantic_native_factorized_residual import verify_factorized_residual
    import tools.verify_semantic_native_residual as residual_verifier

    measured, sources = paths()
    plan = {"sources": sources, "scales": [0., 1.], "training_plan_sha256": "training",
            "candidate_checkpoint_receipt_sha256": "candidate",
            "baseline_checkpoint_receipt_sha256": "base"}
    plan["plan_sha256"] = digest(plan)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    (tmp_path / "rows").mkdir()
    for scale, rows in measured.items():
        for row in rows:
            (tmp_path / "rows" / f"{scale:g}-{row['source']}.json").write_text(
                json.dumps({**row, "receipt_sha256": digest(row)}))
    monkeypatch.setattr(residual_verifier, "verify_residual", lambda *_: {
        "current_implementation_drift": [], "report_receipt_sha256": "measured", "candidate_step": 101})
    report = factor_residual(tmp_path, tmp_path)
    if defect == "selection":
        report["selected_scales"]["reference"] = 1.
    elif defect == "missing_option":
        report["adjudication"].pop()
    elif defect == "authority":
        report["serving_authority"] = True
    elif defect == "sources":
        report["sources"] = ["a"]
    elif defect == "implementation":
        report["implementation"] = {}
    path = tmp_path / "factor.json"
    path.write_text(json.dumps({**report, "receipt_sha256": digest(report)}))
    if defect is None:
        result = verify_factorized_residual(path, tmp_path)
        assert result["artifacts_verified"] and result["combination_count"] == 8
        assert result["selected_totals"]["exact_teacher_paths"] == 2
    else:
        with pytest.raises(ValueError):
            verify_factorized_residual(path, tmp_path)
