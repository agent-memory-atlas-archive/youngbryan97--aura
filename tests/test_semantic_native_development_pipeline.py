"""The larger stage retains completed windows and preserves its micro boundary."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.evaluate_semantic_native_checkpoint import digest
from tools.run_semantic_native_development_windows import (
    ROOT,
    development_jobs,
    development_policy,
    run_windows,
)


def jobs(tmp_path, *, size=125):
    return development_jobs(training_directory=tmp_path / "fit", fit_verification=tmp_path / "fit/verified.json",
        directory=tmp_path / "full", micro_root=tmp_path / "micro", micro_checkout=tmp_path / "frozen-code",
        source_report=tmp_path / "source.json", bundles=["a=/a", "b=/b"], python="/python",
        window_size=size, max_seconds=3600.)


def save(path, body, field="receipt_sha256"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**body, field: digest(body)}))


def executor(pipeline, *, failed_window=None, defect=None):
    calls = []

    def invoke(item):
        calls.append(item["name"])
        command = item["command"]
        if item["name"].endswith("-plan"):
            output = Path(command[command.index("--directory") + 1])
            # The bounds the plan stage freezes from its own command, which a
            # continuation is checked against (cdcd3c830, 29 September).
            bounds = {}
            if "--max-seconds" in command:
                bounds["max_seconds"] = float(command[command.index("--max-seconds") + 1])
            if "--search-nodes" in command:
                bounds["search_nodes"] = int(command[command.index("--search-nodes") + 1])
            if "--prefix-strategy" in command:
                bounds["prefix_strategy"] = command[command.index("--prefix-strategy") + 1]
            if "--decision-score-execution" in command:
                bounds["decision_score_execution"] = command[command.index("--decision-score-execution") + 1]
            save(output / "plan.json", {"training_plan_sha256": "train", "checkpoint_receipt_sha256": "weights",
                **bounds,
                "weight_mode": command[command.index("--weight-mode") + 1],
                "implementation": {"tools/evaluate_semantic_native_grammar.py": hashlib.sha256(
                    (ROOT / "tools/evaluate_semantic_native_grammar.py").read_bytes()).hexdigest()}}, "plan_sha256")
        elif item["name"].endswith("-decode"):
            output = Path(command[command.index("--directory") + 1])
            save(output / "report.json", {"measurement": item["name"]})
        elif item["name"].endswith("-verify"):
            output = Path(command[command.index("--output") + 1])
            save(output, {"artifacts_verified": defect != "unverified",
                          "current_implementation_drift": ["changed"] if defect == "drift" else []})
        elif item["name"].endswith("-compare"):
            window = next(row for row in pipeline["windows"] if row["compare"] == item)
            count = window["count"]
            save(Path(window["directory"]) / "comparison.json", {
                "comparison": {"population": count,
                    "procedure": {"fitted_correct": count - (window["offset"] == failed_window), "regressions": 0},
                    "answer": {"fitted_correct": count, "regressions": 0}},
                "source_intervention": {"population": count,
                    "source_outcomes": [{"arms": {"fitted": {"decode_status": "completed",
                                                              "bound_forced_completion": False}}}] * count}})
        else:
            output = Path(command[command.index("--output") + 1])
            save(output, {"full_development_passed": True})
    return invoke, calls


def run(pipeline, invoke):
    return run_windows(pipeline, training={"plan_sha256": "train"},
                       checkpoint={"receipt_sha256": "weights"}, invoke=invoke)


def test_fixed_policy_covers_all_500_once_under_identical_arm_budgets(tmp_path):
    pipeline = jobs(tmp_path, size=6)
    assert sum(window["count"] for window in pipeline["windows"]) == 500
    assert pipeline["windows"][-1]["offset"] == 498
    assert pipeline["windows"][-1]["count"] == 2
    policy = development_policy(pipeline)
    assert len(policy) == 842
    assert len({tuple(item["command"]) for item in policy}) == len(policy)
    assert all(item["max_invocations"] == 1 for item in policy)
    assert pipeline["micro_replay"]["cwd"] == str(tmp_path / "frozen-code")
    assert "frozen-code/tools/adjudicate_semantic_native_micro_stages.py" in pipeline["micro_replay"]["command"][1]
    for window in pipeline["windows"]:
        for arm in window["arms"]:
            command = arm["decode"]["command"]
            for flag, expected in (("--source-offset", str(window["offset"])),
                                   ("--canary", str(window["count"])), ("--max-seconds", "3600.0"),
                                   ("--max-steps", "8"), ("--search-nodes", "256"),
                                   ("--search-completions", "4"), ("--prefix-strategy", "full")):
                assert command[command.index(flag) + 1] == expected


def test_residual_policy_carries_one_calibration_into_each_nonbase_arm(tmp_path):
    calibration = tmp_path / "calibration"
    pipeline = development_jobs(training_directory=tmp_path / "fit",
        fit_verification=tmp_path / "fit/verified.json", directory=tmp_path / "full",
        micro_root=tmp_path / "micro", micro_checkout=tmp_path / "frozen-code",
        source_report=tmp_path / "source.json", bundles=["a=/a"], python="/python",
        window_size=125, max_seconds=3600., candidate_weight_mode="residual",
        residual_calibration=calibration)
    for window in pipeline["windows"]:
        for arm in window["arms"]:
            command = arm["decode"]["command"]
            if Path(arm["directory"]).name == "base":
                assert command[command.index("--weight-mode") + 1] == "base"
                assert "--residual-calibration" not in command
            else:
                assert command[command.index("--weight-mode") + 1] == "residual"
                assert command[command.index("--residual-calibration") + 1] == str(calibration)
    for command in (pipeline["micro_replay"]["command"], pipeline["adjudicate"]["command"]):
        assert command[command.index("--residual-calibration") + 1] == str(calibration)


@pytest.mark.parametrize("mode,calibration", [("fitted", "present"), ("residual", None),
                                              ("other", None)])
def test_development_policy_rejects_unbound_candidate_modes(tmp_path, mode, calibration):
    with pytest.raises(ValueError, match="calibration binding"):
        development_jobs(training_directory=tmp_path, fit_verification=tmp_path,
            directory=tmp_path, micro_root=tmp_path, micro_checkout=tmp_path,
            source_report=tmp_path, bundles=[], python="/python", window_size=10,
            max_seconds=3600., candidate_weight_mode=mode,
            residual_calibration=None if calibration is None else tmp_path / calibration)


def test_every_plan_is_fixed_before_any_decode_and_full_adjudication_runs_last(tmp_path):
    pipeline = jobs(tmp_path)
    invoke, calls = executor(pipeline)
    assert run(pipeline, invoke) is True
    assert calls[:12] == [item["name"] for item in pipeline["plans"]]
    assert calls[-1] == "development-adjudicate"


@pytest.mark.parametrize("failed_offset", [0, 125, 250, 375])
def test_a_failed_window_preserves_earlier_acceptances_and_stops_the_stage(tmp_path, failed_offset):
    pipeline = jobs(tmp_path)
    invoke, calls = executor(pipeline, failed_window=failed_offset)
    assert run(pipeline, invoke) is False
    assert "development-adjudicate" not in calls
    for window in pipeline["windows"]:
        path = Path(window["directory"]) / "acceptance.json"
        if window["offset"] <= failed_offset:
            body = json.loads(path.read_text())
            assert body["status"] == ("failed" if window["offset"] == failed_offset else "passed")
            assert body["current_stage"] == "full_development"
        else:
            assert not path.exists()
            assert all(arm["decode"]["name"] not in calls for arm in window["arms"])


def test_continuation_never_redecodes_completed_windows(tmp_path):
    pipeline = jobs(tmp_path)
    invoke, _ = executor(pipeline)
    assert run(pipeline, invoke) is True
    acceptance = [Path(window["directory"]) / "acceptance.json" for window in pipeline["windows"]]
    before = [path.read_bytes() for path in acceptance]
    invoke, calls = executor(pipeline)
    assert run(pipeline, invoke) is True
    assert not any(name.endswith(("-plan", "-decode")) for name in calls)
    assert [path.read_bytes() for path in acceptance] == before


@pytest.mark.parametrize("defect", ["unverified", "drift"])
def test_unverified_measurement_cannot_advance_a_window(tmp_path, defect):
    pipeline = jobs(tmp_path)
    invoke, calls = executor(pipeline, defect=defect)
    with pytest.raises(ValueError, match="independent current verification"):
        run(pipeline, invoke)
    assert not any(name.endswith("-compare") for name in calls)


def test_partial_rows_require_an_explicit_attempt_without_repeating_previous_windows(tmp_path):
    pipeline = jobs(tmp_path)
    output = Path(pipeline["windows"][0]["arms"][0]["directory"]) / "rows/source.json"
    save(output, {"partial": True})
    invoke, calls = executor(pipeline)
    with pytest.raises(ValueError, match="new declared attempt"):
        run(pipeline, invoke)
    assert not any(name.endswith("-decode") for name in calls)


@pytest.mark.parametrize("size,seconds", [(0, 3600.), (501, 3600.), (True, 3600.), (6, True),
                                         (6, 0.), (6, float("inf")), (6, 14401.)])
def test_unbounded_or_boolean_policy_inputs_are_rejected(tmp_path, size, seconds):
    with pytest.raises(ValueError, match="fixed finite"):
        development_jobs(training_directory=tmp_path, fit_verification=tmp_path / "verified",
            directory=tmp_path, micro_root=tmp_path, micro_checkout=tmp_path, source_report=tmp_path,
            bundles=[], python="/python", window_size=size, max_seconds=seconds)
