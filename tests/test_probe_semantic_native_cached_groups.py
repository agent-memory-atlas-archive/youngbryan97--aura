"""The real-model cache probe reports observed agreement, not promotion."""

import hashlib

import pytest

from tools.probe_semantic_native_causal_groups import (
    cached_probe_metrics,
    require_source_matched_fit,
)


def case(*, direct=(0.1, 0.2), cached=(0.100001, 0.2), recorded=(0.1, 0.2)):
    return {"direct": direct, "cached": cached, "recorded_scores": recorded,
            "direct_seconds": 3., "shared_seconds": 2., "cached_seconds": 1.,
            "cached_forwards": 2}


def test_cached_probe_reports_bounded_difference_and_measured_work():
    result = cached_probe_metrics([case(), case()])
    assert result["max_direct_cached_score_error"] == pytest.approx(1e-6)
    assert result["cached_winners_match_direct"] is True
    assert result["cached_winners_match_recorded"] is True
    assert result["observed_seconds"] == {"direct": 6., "shared": 4., "cached": 2.}
    assert result["cached_group_forwards"] == 4


def test_cached_probe_exposes_changed_winner_and_incomplete_measurement():
    result = cached_probe_metrics([case(cached=(0.3, 0.2))])
    assert result["cached_winners_match_direct"] is False
    assert result["cached_winners_match_recorded"] is False
    with pytest.raises(ValueError, match="complete"):
        cached_probe_metrics([case(cached=())])


def test_probe_refuses_model_fit_or_residual_source_drift(tmp_path):
    source = tmp_path / "fit.py"
    source.write_text("original", encoding="ascii")
    training = {"implementation": {"fit.py": hashlib.sha256(source.read_bytes()).hexdigest()}}
    residual = {"current_implementation_drift": []}
    require_source_matched_fit(training, residual, root=tmp_path)
    source.write_text("changed", encoding="ascii")
    with pytest.raises(ValueError, match="source-matched"):
        require_source_matched_fit(training, residual, root=tmp_path)
    source.write_text("original", encoding="ascii")
    residual["current_implementation_drift"] = ["residual.py"]
    with pytest.raises(ValueError, match="source-matched"):
        require_source_matched_fit(training, residual, root=tmp_path)
