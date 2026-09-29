"""A small native screen can reject candidates without claiming promotion."""

import json
from pathlib import Path

from tools.evaluate_semantic_native_checkpoint import digest
from tools.run_semantic_native_micro_stages import stage_jobs
from tools.run_semantic_native_reference_screen import (
    reference_screen_jobs,
    run_screen,
    screen_policy,
    screen_verdict,
)


def _jobs(tmp_path):
    jobs = stage_jobs(training_directory=tmp_path / "training",
        fit_verification=tmp_path / "fit.json", directory=tmp_path / "screen",
        source_report=tmp_path / "source.json", bundles=["source=/source"],
        training_plan_sha256="f" * 64, python="/python", search_nodes=16,
        max_seconds=1800., decision_score_execution="causal_groups",
        candidate_weight_mode="residual", residual_calibration=tmp_path / "calibration")
    return reference_screen_jobs(jobs)


def _save(path, body, field):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**body, field: digest(body)}))


def _comparison(*, dependent=1, fitted_correct=3, regressions=0):
    return {"receipt_sha256": "c" * 64,
        "comparison": {"population": 3,
            "procedure": {"fitted_correct": fitted_correct, "regressions": regressions},
            "answer": {"fitted_correct": fitted_correct, "regressions": regressions}},
        "source_intervention": {"population": 3, "source_dependent_gains": dependent,
            "source_outcomes": [{"arms": {"fitted": {"decode_status": "completed",
                "bound_forced_completion": False}}} for _ in range(3)]}}


def test_screen_freezes_only_three_matched_reference_arms(tmp_path):
    plans, arms = _jobs(tmp_path)
    policy = screen_policy(plans, arms)
    assert len(policy) == 9
    assert [Path(arm["directory"]).name for arm in arms] == ["fitted", "base", "erasure"]
    for arm in arms:
        command = arm["decode"]["command"]
        assert command[command.index("--search-nodes") + 1] == "16"
        assert command[command.index("--max-seconds") + 1] == "1800"
        assert command[command.index("--decision-score-execution") + 1] == "causal_groups"
        assert arm["decode"]["max_invocations"] == 1


def test_screen_pass_requires_full_cohort_and_source_dependent_gain():
    assert screen_verdict(_comparison()) is True
    assert screen_verdict(_comparison(dependent=0)) is False
    assert screen_verdict(_comparison(fitted_correct=2)) is False
    assert screen_verdict(_comparison(regressions=1)) is False


def test_screen_reuses_complete_reports_but_reverifies_before_comparison(tmp_path):
    plans, arms = _jobs(tmp_path)
    calls = []
    training = {"plan_sha256": "f" * 64}
    candidate = {"receipt_sha256": "c" * 64}
    baseline = {"receipt_sha256": "b" * 64}

    def invoke(item):
        calls.append(item["name"])
        command = item["command"]
        directory = Path(command[command.index("--directory") + 1]) if "--directory" in command else None
        if item["name"].endswith("-plan"):
            body = {"training_plan_sha256": training["plan_sha256"],
                    "checkpoint_receipt_sha256": baseline["receipt_sha256"] if
                    "--weight-mode" in command and command[command.index("--weight-mode") + 1] == "base"
                    else candidate["receipt_sha256"],
                    "weight_mode": command[command.index("--weight-mode") + 1],
                    "max_seconds": 1800., "search_nodes": 16,
                    "decision_score_execution": "causal_groups", "implementation": {},
                    "residual_calibration": {"directory": str((tmp_path / "calibration").resolve()),
                        "report_receipt_sha256": "r" * 64} if
                        command[command.index("--weight-mode") + 1] == "residual" else None}
            _save(directory / "plan.json", body, "plan_sha256")
        elif item["name"].endswith("-decode"):
            _save(directory / "report.json", {"complete": True}, "receipt_sha256")
        else:
            output = Path(command[command.index("--output") + 1])
            _save(output, {"artifacts_verified": True, "current_implementation_drift": []},
                  "receipt_sha256")

    def execute():
        return run_screen(plans, arms, training=training, checkpoint=candidate,
            baseline_checkpoint=baseline, residual_report_receipt_sha256="r" * 64,
            invoke=invoke, compare=_comparison)

    assert execute()["screen_passed"] is True
    assert sum(name.endswith("-decode") for name in calls) == 3
    calls.clear()
    assert execute()["screen_passed"] is True
    assert not any(name.endswith("-decode") for name in calls)
    assert sum(name.endswith("-plan") for name in calls) == 3
    assert sum(name.endswith("-verify") for name in calls) == 3
