"""The micro controller advances only verified stages and never repeats a decode."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.evaluate_semantic_native_checkpoint import digest
from tools.run_semantic_native_micro_stages import (
    ROOT,
    broker_policy,
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
