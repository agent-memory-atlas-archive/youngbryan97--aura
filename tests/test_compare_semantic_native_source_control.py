"""A source-content intervention must be the only difference between arms."""

import pytest

from tools.compare_semantic_native_source_control import matched_source_control


def _arms():
    common = {"sources": ["a", "b"], "weight_mode": "fitted",
              "max_steps": 3, "checkpoint_receipt_sha256": "checkpoint"}
    full = {**common, "source_evidence": "source_text", "plan_sha256": "full-plan"}
    erased = {**common, "source_evidence": "source_token_erasure",
              "plan_sha256": "erased-plan"}
    full_report = {"plan_sha256": "full-plan", "population": 2,
                   "pair_exact": 1, "source_responsive": 1}
    erased_report = {"plan_sha256": "erased-plan", "population": 2,
                     "pair_exact": 0, "source_responsive": 0}
    full_rows = tuple({"source_sha256": source, "source_evidence": "source_text",
                       "program_equivalent": True, "answer_correct": True,
                       "program": {"steps": [source]}} for source in common["sources"])
    erased_rows = tuple({"source_sha256": source,
                         "source_evidence": "source_token_erasure",
                         "program_equivalent": source == "b", "answer_correct": source == "b",
                         "program": {"steps": ["a"]}}
                        for source in common["sources"])
    return full, erased, full_report, erased_report, full_rows, erased_rows


def test_adjudicator_counts_both_causal_changes_and_correctness():
    result = matched_source_control(*_arms())
    assert result["population"] == 2
    assert result["full_correct"] == 2 and result["erased_correct"] == 1
    assert result["full_only"] == 1 and result["erased_only"] == 0
    assert result["program_changed"] == 1
    assert result["full_source_responsive"] == 1
    assert result["erased_source_responsive"] == 0


def test_adjudicator_rejects_model_budget_population_and_mode_drift():
    full, erased, full_report, erased_report, full_rows, erased_rows = _arms()
    for changed in ({**erased, "max_steps": 4},
                    {**erased, "checkpoint_receipt_sha256": "other"},
                    {**erased, "source_evidence": "source_text"}):
        with pytest.raises(ValueError):
            matched_source_control(full, changed, full_report, erased_report,
                                   full_rows, erased_rows)
    with pytest.raises(ValueError, match="population"):
        matched_source_control(full, erased, full_report,
                               {**erased_report, "population": 1}, full_rows, erased_rows)
    with pytest.raises(ValueError, match="row coverage"):
        matched_source_control(full, erased, full_report, erased_report,
                               full_rows, erased_rows[::-1])
