"""The micro controller advances only verified stages and never repeats a decode."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.evaluate_semantic_native_checkpoint import digest
from tools.run_semantic_native_micro_stages import (
    ROOT,
    broker_policy,
    check_existing_plan,
    fitted_stage_blockers,
    require_learned_checkpoint,
    run_stages,
    stage_jobs,
    wait_for_fit,
)


def fixture(tmp_path):
    training = {"plan_sha256": "f" * 64}
    checkpoint = {"receipt_sha256": "c" * 64}
    jobs = stage_jobs(training_directory=tmp_path / "training",
        fit_verification=tmp_path / "fit-verification.json", directory=tmp_path / "micro",
        source_report=tmp_path / "source.json", bundles=["source=/source", "counterfactual=/counterfactual"],
        training_plan_sha256=training["plan_sha256"], python="/python")
    return jobs, training, checkpoint


def save(path, body, field="receipt_sha256"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**body, field: digest(body)}))


def executor(jobs, training, checkpoint, *, failed_stage=None, verification_defect=None):
    calls = []

    def invoke(item):
        calls.append(item["name"])
        command = item["command"]
        if item["name"].endswith("-plan"):
            directory = Path(command[command.index("--directory") + 1])
            implementation = {"tools/evaluate_semantic_native_grammar.py": hashlib.sha256(
                (ROOT / "tools/evaluate_semantic_native_grammar.py").read_bytes()).hexdigest()}
            save(directory / "plan.json", {"training_plan_sha256": training["plan_sha256"],
                "checkpoint_receipt_sha256": checkpoint["receipt_sha256"],
                "weight_mode": command[command.index("--weight-mode") + 1],
                "max_seconds": float(command[command.index("--max-seconds") + 1]),
                "search_nodes": int(command[command.index("--search-nodes") + 1]),
                "implementation": implementation}, "plan_sha256")
        elif item["name"].endswith("-decode"):
            directory = Path(command[command.index("--directory") + 1])
            save(directory / "report.json", {"fixture": item["name"]})
        elif item["name"].endswith("-verify"):
            output = Path(command[command.index("--output") + 1])
            save(output, {"artifacts_verified": verification_defect != "unverified",
                          "current_implementation_drift": ["changed"] if verification_defect == "drift" else []})
        else:
            stage = next(row for row in jobs["stages"] if row["adjudicate"] == item)
            index = jobs["stages"].index(stage)
            statuses = [{"stage": row["stage"], "status": "passed" if ordinal <= index else "not_run"}
                        for ordinal, row in enumerate(jobs["stages"])]
            if stage["stage"] == failed_stage:
                statuses[index]["status"] = "failed"
            save(Path(stage["progress_path"]), {"stages": statuses,
                "full_development_ready": index == 2 and failed_stage is None})

    return invoke, calls


def test_every_arm_has_the_same_frozen_budget_and_broker_invocation_bound(tmp_path):
    jobs, _, _ = fixture(tmp_path)
    policy = broker_policy(jobs)
    assert len(policy) == 24
    assert len({tuple(row["command"]) for row in policy}) == len(policy)
    assert all(row["max_invocations"] == 1 for row in policy)
    assert [row["stage"] for row in jobs["stages"]] == [
        "reference_requests", "relation_controls", "retained_requests"]
    for stage in jobs["stages"]:
        for arm in stage["arms"]:
            command = arm["decode"]["command"]
            for flag, value in (("--max-steps", "8"), ("--search-completions", "4"),
                                ("--search-nodes", "256"), ("--prefix-strategy", "full"),
                                ("--max-seconds", "3600"), ("--search-score-mode", "native_nonpositive")):
                assert command[command.index(flag) + 1] == value
            if stage["stage"] == "retained_requests":
                assert "--seed" not in command
                assert command.count("--bundle") == 2
            else:
                assert command[command.index("--seed") + 1] == str(int("ffffffff", 16))


def test_early_fitted_rejection_skips_expensive_controls_and_preserves_evidence(tmp_path):
    jobs, training, checkpoint = fixture(tmp_path)
    jobs["fitted_rejection_policy"] = "reject_after_verified_fitted_v1"
    original, calls = executor(jobs, training, checkpoint)
    def invoke(item):
        original(item)
        if item["name"].endswith("-verify"):
            output = Path(item["command"][item["command"].index("--output") + 1])
            save(output, {"artifacts_verified": True, "current_implementation_drift": [],
                "totals": {"population": 3, "answer_correct": 2, "program_equivalent": 1,
                           "bound_forced_completion": 0}})
    result = run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke)
    assert result["stages"] == [{"stage": "reference_requests", "status": "failed"},
        {"stage": "relation_controls", "status": "not_run"},
        {"stage": "retained_requests", "status": "not_run"}]
    assert result["comparison_arms_skipped"] is True
    assert result["source_attribution_measured"] is False and result["full_development_ready"] is False
    assert not any(name.endswith("-decode") and "fitted" not in name for name in calls)
    assert [row["metric"] for row in result["blockers"]] == ["answer_correct", "program_equivalent"]
    before = list(calls)
    replay = run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke)
    assert replay == result and calls[len(before):] == ["reference-fitted-verify"]


def test_perfect_fitted_arm_cannot_skip_baseline_or_erasure(tmp_path):
    jobs, training, checkpoint = fixture(tmp_path)
    jobs["fitted_rejection_policy"] = "reject_after_verified_fitted_v1"
    original, calls = executor(jobs, training, checkpoint)
    def invoke(item):
        original(item)
        if item["name"].endswith("-verify") and "-fitted-" in item["name"]:
            population = 3 if item["name"].startswith("reference") else 6
            output = Path(item["command"][item["command"].index("--output") + 1])
            save(output, {"artifacts_verified": True, "current_implementation_drift": [],
                "totals": {"population": population, "answer_correct": population,
                           "program_equivalent": population, "bound_forced_completion": 0}})
    result = run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke)
    assert result["full_development_ready"] is True
    assert "retained-base-decode" in calls and "retained-erasure-decode" in calls


@pytest.mark.parametrize("defect", ["partial", "unverified", "drift", "boolean", "overflow", "missing"])
def test_fitted_screen_refuses_unmeasured_or_invalid_successes(defect):
    verification = {"artifacts_verified": True, "current_implementation_drift": [],
        "totals": {"population": 6, "answer_correct": 6, "program_equivalent": 6,
                   "bound_forced_completion": 0}}
    assert fitted_stage_blockers(verification, population=6) == []
    if defect == "partial":
        verification["totals"]["population"] = 5
    elif defect == "unverified":
        verification["artifacts_verified"] = False
    elif defect == "drift":
        verification["current_implementation_drift"] = ["changed"]
    elif defect == "boolean":
        verification["totals"]["answer_correct"] = True
    elif defect == "overflow":
        verification["totals"]["answer_correct"] = 7
    else:
        verification["totals"].pop("program_equivalent")
    with pytest.raises(ValueError, match="verified complete totals"):
        fitted_stage_blockers(verification, population=6)


def test_micro_policy_freezes_explicit_longer_bound_into_every_arm(tmp_path):
    jobs = stage_jobs(training_directory=tmp_path / "training",
        fit_verification=tmp_path / "fit-verification.json", directory=tmp_path / "micro",
        source_report=tmp_path / "source.json", bundles=["source=/source"],
        training_plan_sha256="f" * 64, python="/python", max_seconds=14400)
    for stage in jobs["stages"]:
        for arm in stage["arms"]:
            command = arm["decode"]["command"]
            assert command[command.index("--max-seconds") + 1] == "14400"
            assert arm["decode"]["timeout_s_max"] == 14700
    assert all(item["max_invocations"] == 1 for item in broker_policy(jobs))


def test_micro_policy_isolates_opt_in_causal_execution(tmp_path):
    jobs = stage_jobs(training_directory=tmp_path / "training",
        fit_verification=tmp_path / "fit-verification.json", directory=tmp_path / "micro",
        source_report=tmp_path / "source.json", bundles=["source=/source"],
        training_plan_sha256="f" * 64, python="/python",
        decision_score_execution="causal_groups")
    for stage in jobs["stages"]:
        for arm in stage["arms"]:
            command = arm["decode"]["command"]
            assert command[command.index("--decision-score-execution") + 1] == "causal_groups"
    with pytest.raises(ValueError, match="decision score execution"):
        stage_jobs(training_directory=tmp_path / "training",
            fit_verification=tmp_path / "fit-verification.json", directory=tmp_path / "micro",
            source_report=tmp_path / "source.json", bundles=["source=/source"],
            training_plan_sha256="f" * 64, python="/python",
            decision_score_execution="unknown")


def test_micro_policy_binds_cached_grouped_search(tmp_path):
    kwargs = dict(training_directory=tmp_path / "training",
                  fit_verification=tmp_path / "fit-verification.json", directory=tmp_path / "micro",
                  source_report=tmp_path / "source.json", bundles=["source=/source"],
                  training_plan_sha256="f" * 64, python="/python")
    jobs = stage_jobs(**kwargs, decision_score_execution="causal_groups", prefix_strategy="trie")
    for stage in jobs["stages"]:
        for arm in stage["arms"]:
            command = arm["decode"]["command"]
            assert command[command.index("--prefix-strategy") + 1] == "trie"
    with pytest.raises(ValueError, match="cached prefix"):
        stage_jobs(**kwargs, prefix_strategy="trie")
    saved = tmp_path / "plan.json"
    save(saved, {"training_plan_sha256": "f" * 64,
                 "checkpoint_receipt_sha256": "c" * 64,
                 "weight_mode": "fitted", "prefix_strategy": "trie",
                 "implementation": {}}, "plan_sha256")
    with pytest.raises(ValueError, match="prefix strategy"):
        check_existing_plan(saved, training={"plan_sha256": "f" * 64},
                            checkpoint={"receipt_sha256": "c" * 64},
                            command=["--weight-mode", "fitted", "--prefix-strategy", "full"])


@pytest.mark.parametrize("bound", [0, -1, 14401, float("inf"), float("nan"), True])
def test_micro_policy_refuses_unbounded_or_invalid_decode_time(tmp_path, bound):
    with pytest.raises(ValueError, match="runtime bound"):
        stage_jobs(training_directory=tmp_path / "training",
            fit_verification=tmp_path / "fit-verification.json", directory=tmp_path / "micro",
            source_report=tmp_path / "source.json", bundles=["source=/source"],
            training_plan_sha256="f" * 64, python="/python", max_seconds=bound)


def test_micro_resume_refuses_a_different_decode_budget(tmp_path):
    saved = tmp_path / "plan.json"
    save(saved, {"training_plan_sha256": "train", "checkpoint_receipt_sha256": "candidate",
                 "weight_mode": "fitted", "max_seconds": 3600., "search_nodes": 256,
                 "implementation": {}}, "plan_sha256")
    check_existing_plan(saved, training={"plan_sha256": "train"},
                        checkpoint={"receipt_sha256": "candidate"},
                        command=["--weight-mode", "fitted", "--max-seconds", "3600",
                                 "--search-nodes", "256"])
    with pytest.raises(ValueError, match="runtime bound"):
        check_existing_plan(saved, training={"plan_sha256": "train"},
                            checkpoint={"receipt_sha256": "candidate"},
                            command=["--weight-mode", "fitted", "--max-seconds", "14400"])
    with pytest.raises(ValueError, match="search bound"):
        check_existing_plan(saved, training={"plan_sha256": "train"},
                            checkpoint={"receipt_sha256": "candidate"},
                            command=["--weight-mode", "fitted", "--search-nodes", "16"])


def test_residual_micro_policy_uses_one_source_calibration_for_every_candidate_arm(tmp_path):
    calibration = tmp_path / "calibration"
    jobs = stage_jobs(training_directory=tmp_path / "training",
        fit_verification=tmp_path / "fit-verification.json", directory=tmp_path / "micro",
        source_report=tmp_path / "source.json", bundles=["source=/source"],
        training_plan_sha256="f" * 64, python="/python", candidate_weight_mode="residual",
        residual_calibration=calibration)
    for stage in jobs["stages"]:
        for arm in stage["arms"]:
            command = arm["decode"]["command"]
            if Path(arm["directory"]).name == "base":
                assert command[command.index("--weight-mode") + 1] == "base"
                assert "--residual-calibration" not in command
            else:
                assert command[command.index("--weight-mode") + 1] == "residual"
                assert command[command.index("--residual-calibration") + 1] == str(calibration)
        command = stage["adjudicate"]["command"]
        assert command[command.index("--residual-calibration") + 1] == str(calibration)


def test_residual_resume_checks_baseline_and_candidate_receipts_before_decode(tmp_path):
    training = {"plan_sha256": "train"}
    baseline = {"receipt_sha256": "baseline"}
    candidate = {"receipt_sha256": "candidate"}
    base = tmp_path / "base.json"
    save(base, {"training_plan_sha256": "train", "checkpoint_receipt_sha256": "baseline",
                "weight_mode": "base", "implementation": {}}, "plan_sha256")
    check_existing_plan(base, training=training, checkpoint=candidate,
        baseline_checkpoint=baseline, command=["--weight-mode", "base"])
    with pytest.raises(ValueError, match="candidate or code"):
        check_existing_plan(base, training=training, checkpoint=candidate,
            command=["--weight-mode", "base"])
    calibration = tmp_path / "calibration"
    residual = tmp_path / "residual.json"
    save(residual, {"training_plan_sha256": "train", "checkpoint_receipt_sha256": "candidate",
        "weight_mode": "residual", "implementation": {},
        "residual_calibration": {"directory": str(calibration.resolve()),
                                 "report_receipt_sha256": "measured"}}, "plan_sha256")
    command = ["--weight-mode", "residual", "--residual-calibration", str(calibration)]
    check_existing_plan(residual, training=training, checkpoint=candidate, command=command,
        baseline_checkpoint=baseline, residual_report_receipt_sha256="measured")
    with pytest.raises(ValueError, match="calibration"):
        check_existing_plan(residual, training=training, checkpoint=candidate, command=command,
            baseline_checkpoint=baseline, residual_report_receipt_sha256="other")


@pytest.mark.parametrize("mode,calibration", [("fitted", "present"), ("residual", None),
                                              ("other", None)])
def test_micro_policy_rejects_unbound_candidate_modes(tmp_path, mode, calibration):
    with pytest.raises(ValueError, match="calibration binding"):
        stage_jobs(training_directory=tmp_path, fit_verification=tmp_path,
            directory=tmp_path, source_report=tmp_path, bundles=[], training_plan_sha256="f" * 64,
            python="/python", candidate_weight_mode=mode,
            residual_calibration=None if calibration is None else tmp_path / calibration)


def test_all_plans_are_frozen_before_decode_and_stages_advance_immediately(tmp_path):
    jobs, training, checkpoint = fixture(tmp_path)
    invoke, calls = executor(jobs, training, checkpoint)
    result = run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke)
    assert result["full_development_ready"] is True
    assert calls[:7] == [item["name"] for item in jobs["plans"]]
    for index in range(2):
        current = jobs["stages"][index]["adjudicate"]["name"]
        following = jobs["stages"][index + 1]["arms"][0]["decode"]["name"]
        assert calls.index(following) == calls.index(current) + 1


@pytest.mark.parametrize("index", range(3))
def test_a_failed_acceptance_preserves_earlier_passes_and_does_not_run_later_decodes(tmp_path, index):
    jobs, training, checkpoint = fixture(tmp_path)
    failed = jobs["stages"][index]["stage"]
    invoke, calls = executor(jobs, training, checkpoint, failed_stage=failed)
    result = run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke)
    assert [row["status"] for row in result["stages"]] == (
        ["passed"] * index + ["failed"] + ["not_run"] * (2 - index))
    for stage in jobs["stages"][index + 1:]:
        assert all(arm["decode"]["name"] not in calls for arm in stage["arms"])


def test_completed_decodes_are_reverified_on_cpu_but_never_redecoded(tmp_path):
    jobs, training, checkpoint = fixture(tmp_path)
    invoke, calls = executor(jobs, training, checkpoint)
    run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke)
    calls.clear()
    result = run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke)
    assert result["full_development_ready"] is True
    assert len(calls) == 10
    assert all(name.endswith(("-verify", "-adjudicate")) for name in calls)


@pytest.mark.parametrize("defect", ["weights", "code", "partial", "unverified", "drift"])
def test_changed_or_partial_artifacts_cannot_be_resumed_as_passes(tmp_path, defect):
    jobs, training, checkpoint = fixture(tmp_path)
    invoke, calls = executor(jobs, training, checkpoint,
                             verification_defect=defect if defect in {"unverified", "drift"} else None)
    if defect in {"weights", "code", "partial"}:
        for item in jobs["plans"]:
            invoke(item)
        calls.clear()
        if defect == "weights":
            checkpoint = {"receipt_sha256": "other"}
        elif defect == "code":
            path = Path(jobs["stages"][0]["arms"][0]["directory"]) / "plan.json"
            body = json.loads(path.read_bytes())
            body.pop("plan_sha256")
            body["implementation"]["tools/evaluate_semantic_native_grammar.py"] = "changed"
            save(path, body, "plan_sha256")
        else:
            output = Path(jobs["stages"][0]["arms"][0]["directory"])
            save(output / "rows/one.json", {"fixture": "partial"})
    with pytest.raises(ValueError):
        run_stages(jobs, training=training, checkpoint=checkpoint, invoke=invoke)
    assert jobs["stages"][1]["arms"][0]["decode"]["name"] not in calls
    if defect in {"weights", "code", "partial"}:
        assert all(not name.endswith("-decode") for name in calls)


def terminal():
    return {"terminal": True, "child_state": "dead", "receipt": {
        "passed": True, "containment_verified": True, "timed_out": False, "returncode": 0}}


def test_micro_rejects_an_unfitted_selected_checkpoint_before_decode():
    with pytest.raises(ValueError, match="selected learned checkpoint"):
        require_learned_checkpoint({"step": 0})
    with pytest.raises(ValueError, match="selected learned checkpoint"):
        require_learned_checkpoint({"step": None})
    require_learned_checkpoint({"step": 101})


def test_fit_wait_advances_only_after_the_existing_supervisor_proves_success():
    states = iter([{"terminal": False, "supervisor_alive": True, "state": "running"}, terminal()])
    waits = []
    result = wait_for_fit(Path("/fit/supervisor"), timeout=60., inspect=lambda path: next(states),
                         clock=lambda: 0., sleep=waits.append)
    assert result == terminal()["receipt"]
    assert waits == [30.]


@pytest.mark.parametrize("defect", ["failed", "uncontained", "timeout", "alive", "lost", "unknown", "expired"])
def test_fit_failure_or_unknown_ownership_never_advances(defect):
    state = terminal()
    times = iter([0., 60.])
    if defect == "failed":
        state["receipt"]["passed"] = False
    elif defect == "uncontained":
        state["receipt"]["containment_verified"] = False
    elif defect == "timeout":
        state["receipt"]["timed_out"] = True
    elif defect == "alive":
        state["child_state"] = "alive"
    else:
        state = {"terminal": False, "supervisor_alive": defect != "lost", "state": "running",
                 "completion_indeterminate": defect == "unknown"}
    with pytest.raises(ValueError):
        wait_for_fit(Path("/fit/supervisor"), timeout=30., inspect=lambda path: state,
                     clock=lambda: next(times), sleep=lambda duration: None)
