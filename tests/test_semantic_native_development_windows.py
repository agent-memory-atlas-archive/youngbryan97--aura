"""Fixed development windows preserve coverage, paired budgets, and prior work."""

import copy
import hashlib
from types import SimpleNamespace

import pytest

from tools.adjudicate_semantic_native_development import complete_development
from tools.evaluate_semantic_native_checkpoint import digest
from tools.semantic_native_retained_sources import retained_native_source_window
from tools.verify_semantic_native_grammar import (
    verified_dataset,
    verified_examples,
    verified_input_grounding,
    verified_prefix_execution,
    verified_source_window,
    verified_weight_mode,
)


def population(monkeypatch):
    examples = tuple(SimpleNamespace(source_text=f"source-{index}") for index in range(500))
    basis = {"split": "validation", "source_report_path": "/report",
             "source_manifest_paths": {"a": "/a"}}
    calls = []

    def load(path, bundles, *, split, count):
        calls.append((path, bundles, split, count))
        return examples[:count], basis

    monkeypatch.setattr("tools.semantic_native_retained_sources.load_retained_native_sources", load)
    return examples, basis, calls


def test_windows_partition_the_entire_order_without_ranking_or_label_selection(monkeypatch):
    examples, basis, calls = population(monkeypatch)
    observed = []
    for offset in range(0, 500, 7):
        count = min(7, 500 - offset)
        batch, reconstructed, window = retained_native_source_window(
            "/report", ["a=/a"], split="validation", offset=offset, count=count)
        assert reconstructed == basis
        assert window == {"offset": offset, "count": count, "population": 500,
                          "ordered_sources_sha256": digest([
                              hashlib.sha256(item.source_text.encode()).hexdigest() for item in examples])}
        observed.extend(batch)
    assert tuple(observed) == examples
    assert all(count == 500 for _, _, _, count in calls)


@pytest.mark.parametrize("offset,count", [(-1, 1), (500, 1), (499, 2), (0, 0), (True, 1), (0, True)])
def test_invalid_windows_fail_before_loading_sources(offset, count):
    with pytest.raises(ValueError, match="complete population"):
        retained_native_source_window("/unused", (), split="validation", offset=offset, count=count)


def window_plan(sources, *, offset=0, population_sha=None):
    return {"schema": "aura.semantic_native_grammar_plan.v12", "dataset": "retained_validation",
            "weight_mode": "fitted", "source_evidence": "source_text", "seed": 0,
            "training_plan_sha256": "train", "checkpoint_receipt_sha256": "checkpoint",
            "source_cohort_basis": {"split": "validation"}, "sources": sources,
            "plan_sha256": "plan", "max_steps": 8, "max_seconds": 3600.,
            "search_nodes": 256, "search_completions": 4,
            "input_grounding": "semantic_public_character_inputs.v1",
            "source_window": {"offset": offset, "count": len(sources), "population": 500,
                              "ordered_sources_sha256": population_sha or digest(sources)}}


def test_new_schema_binds_window_without_changing_historical_full_prefix_contract():
    plan = window_plan(["source"])
    report = {**plan, "schema": "aura.semantic_native_grammar.v12"}
    assert verified_weight_mode(plan, report) == "fitted"
    assert verified_dataset(plan, report) == ("retained_validation", 0)
    assert verified_input_grounding(plan, report) == "semantic_public_character_inputs.v1"
    assert verified_prefix_execution(plan, report) is None
    assert verified_source_window(plan, report) == plan["source_window"]
    with pytest.raises(ValueError, match="source window"):
        verified_dataset(plan, {**report, "source_window": {**plan["source_window"], "offset": 1}})
    with pytest.raises(ValueError, match="historical"):
        verified_dataset({**plan, "schema": "aura.semantic_native_grammar_plan.v6"}, report)


def test_cached_grouped_window_keeps_complete_population_binding():
    plan = window_plan(["source"])
    plan.update(schema="aura.semantic_native_grammar_plan.v15", prefix_strategy="trie",
                decision_score_execution="causal_groups")
    report = {**plan, "schema": "aura.semantic_native_grammar.v15"}
    assert verified_weight_mode(plan, report) == "fitted"
    assert verified_dataset(plan, report) == ("retained_validation", 0)
    assert verified_source_window(plan, report) == plan["source_window"]
    assert verified_prefix_execution(plan, report) is None


def test_window_invariant_is_executable_and_detects_a_permissive_validator(monkeypatch):
    from tools.verify_semantic_native_grammar import _native_development_window_contract

    assert list(_native_development_window_contract()) == []
    monkeypatch.setattr("tools.verify_semantic_native_grammar.verified_source_window",
                        lambda plan, report: plan["source_window"])
    assert len(list(_native_development_window_contract())) == 3


def test_verifier_reconstructs_exact_offset_from_the_complete_source_inventory(monkeypatch):
    examples, basis, _ = population(monkeypatch)
    sources = [hashlib.sha256(item.source_text.encode()).hexdigest() for item in examples]
    plan = window_plan(sources[10:15], offset=10, population_sha=digest(sources))
    plan["source_cohort_basis"] = basis
    assert verified_examples(plan, dataset="retained_validation", seed=0) == examples[10:15]
    plan["source_window"]["ordered_sources_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="reconstruction"):
        verified_examples(plan, dataset="retained_validation", seed=0)


def measurements(*, size=20, mode="fitted"):
    sources = [hashlib.sha256(f"source-{index}".encode()).hexdigest() for index in range(500)]
    windows = []
    for offset in range(0, 500, size):
        batch = sources[offset:offset + size]
        plan = window_plan(batch, offset=offset, population_sha=digest(sources))
        if mode == "residual":
            plan["schema"] = "aura.semantic_native_grammar_plan.v13"
            plan["weight_mode"] = "residual"
            plan["residual_calibration"] = {"selected_scale": 0.125,
                "report_receipt_sha256": "calibration", "source_only": True}
        measured, intervention = [], []
        for source in batch:
            gained = source == sources[0]
            measured.append({"source_sha256": source, "fitted_procedure": True,
                             "fitted_answer": True, "base_procedure": not gained,
                             "base_answer": not gained})
            intervention.append({"source_sha256": source, "gain_changes_under_erasure": gained,
                "arms": {"fitted": {"decode_status": "completed", "bound_forced_completion": False}}})
        comparison = {"training_plan_sha256": "train", "checkpoint_receipt_sha256": "checkpoint",
                      "receipt_sha256": f"comparison-{offset}",
                      "comparison": {"population": len(batch), "candidate_weight_mode": mode,
                                     "source_outcomes": measured},
                      "source_intervention": {"population": len(batch), "source_outcomes": intervention}}
        windows.append((plan, comparison))
    return windows


def test_all_500_are_required_and_exact_before_advancing_to_fresh_transfer():
    windows = measurements(size=7)
    result = complete_development(windows)
    assert result["full_development_passed"] is True
    assert result["population"] == 500
    assert result["windows"] == 72
    assert result["procedure"] == {"fitted_correct": 500, "base_correct": 499,
                                   "gains": 1, "regressions": 0}
    assert result["answer"] == result["procedure"]
    assert result["source_dependent_gains"] == 1
    assert result["current_stage"] == "fresh_transfer"
    assert result["general_transfer_proven"] is False
    assert result["broad_gain_proven"] is False
    assert result["serving_authority"] is False
    assert result["comparison_receipts"] == [comparison["receipt_sha256"] for _, comparison in windows]


def test_residual_development_is_bound_to_one_measured_candidate_and_still_needs_500():
    windows = measurements(size=7, mode="residual")
    result = complete_development(windows)
    assert result["full_development_passed"] is True
    assert result["candidate_weight_mode"] == "residual"
    assert result["residual_calibration_report_receipt_sha256"] == "calibration"
    windows[1][0]["residual_calibration"]["report_receipt_sha256"] = "other"
    with pytest.raises(ValueError, match="differs"):
        complete_development(windows)


def test_cached_grouped_development_cannot_mix_prefix_execution_windows():
    windows = measurements(size=100, mode="residual")
    for plan, _comparison in windows:
        plan.update(schema="aura.semantic_native_grammar_plan.v15",
                    prefix_strategy="trie", decision_score_execution="causal_groups")
    assert complete_development(windows)["full_development_passed"] is True
    windows[1][0]["prefix_strategy"] = "full"
    with pytest.raises(ValueError, match="differs"):
        complete_development(windows)


def test_residual_window_cannot_inherit_fitted_comparison_or_zero_scale():
    windows = measurements(mode="residual")
    windows[1][1]["comparison"]["candidate_weight_mode"] = "fitted"
    with pytest.raises(ValueError, match="differs"):
        complete_development(windows)
    windows = measurements(mode="residual")
    windows[0][0]["residual_calibration"]["selected_scale"] = 0.
    with pytest.raises(ValueError, match="calibration"):
        complete_development(windows)


@pytest.mark.parametrize("defect", ["missing", "reordered", "overlap", "budget", "weights", "population",
                                   "source", "paired_population"])
def test_missing_or_mismatched_windows_never_become_a_500_request_result(defect):
    windows = copy.deepcopy(measurements())
    if defect == "missing":
        windows.pop()
    elif defect == "reordered":
        windows[0], windows[1] = windows[1], windows[0]
    elif defect == "overlap":
        windows[1][0]["source_window"]["offset"] = 0
    elif defect == "budget":
        windows[1][0]["search_nodes"] += 1
    elif defect == "weights":
        windows[1][0]["checkpoint_receipt_sha256"] = "other"
    elif defect == "population":
        windows[1][0]["source_window"]["ordered_sources_sha256"] = "0" * 64
    elif defect == "source":
        windows[1][1]["comparison"]["source_outcomes"][0]["source_sha256"] = "other"
    else:
        windows[1][1]["source_intervention"]["population"] -= 1
    with pytest.raises(ValueError, match="differs|500 distinct"):
        complete_development(windows)


@pytest.mark.parametrize("defect", ["procedure", "answer", "forced", "incomplete", "no_source_gain"])
def test_a_failed_full_stage_keeps_its_stage_and_does_not_grant_transfer(defect):
    windows = measurements()
    if defect in {"procedure", "answer"}:
        windows[-1][1]["comparison"]["source_outcomes"][-1]["fitted_" + defect] = False
    elif defect == "forced":
        windows[-1][1]["source_intervention"]["source_outcomes"][-1]["arms"]["fitted"][
            "bound_forced_completion"] = True
    elif defect == "incomplete":
        windows[-1][1]["source_intervention"]["source_outcomes"][-1]["arms"]["fitted"][
            "decode_status"] = "search_without_completion"
    else:
        windows[0][1]["source_intervention"]["source_outcomes"][0]["gain_changes_under_erasure"] = False
    result = complete_development(windows)
    assert result["full_development_passed"] is False
    assert result["current_stage"] == "full_development"
    assert len(result["comparison_receipts"]) == len(windows)


@pytest.mark.parametrize("dataset,offset,count,prefix,mode", [
    ("natural_request", 0, 3, "full", "fitted"),
    ("retained_validation", -1, 3, "full", "fitted"),
    ("retained_validation", 499, 3, "full", "fitted"),
    ("retained_validation", 0, 3, "trie", "fitted"),
    ("retained_validation", 0, 3, "full", "residual"),
])
def test_cli_rejects_windows_that_would_change_the_common_computation(
        monkeypatch, dataset, offset, count, prefix, mode):
    from tools.evaluate_semantic_native_grammar import main

    monkeypatch.setattr("sys.argv", ["evaluate", "--training-directory", "/unused", "--directory", "/unused",
        "--dataset", dataset, "--source-offset", str(offset), "--canary", str(count),
        "--prefix-strategy", prefix, "--weight-mode", mode])
    with pytest.raises(SystemExit) as failure:
        main()
    assert failure.value.code == 2
