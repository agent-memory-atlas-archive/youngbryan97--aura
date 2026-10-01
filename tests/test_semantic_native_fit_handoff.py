"""Prepared training advances without repeating it or changing its protocol."""

import copy

import pytest

from tools.probe_semantic_proposer_crossfit import _digest, _save_if_absent
from core.learning.semantic_fit_checkpoint import fit_identity
from tools.run_semantic_native_fit_handoff import (
    fit_jobs, preparation_paths, verified_preparation_document, verify_preparation,
)
from tools.run_semantic_source_handoff import native_preparation_jobs


def _receipt(body, field="receipt_sha256", *, digest=_digest):
    body.pop(field, None)
    return {**body, field: digest(body)}


def _protocol(tmp_path):
    paths = {"candidate": tmp_path / "parent.json", "report": tmp_path / "source.json",
             "bundles": ["one=" + str(tmp_path / "one"), "two=" + str(tmp_path / "two")]}
    jobs = native_preparation_jobs(paths, tmp_path / "bank", tmp_path / "native", tmp_path,
                                   python="python")
    return _receipt({"schema": "aura.semantic_source_native_preparation_plan.v1", "jobs": jobs},
                    digest=fit_identity)


def test_fit_keeps_all_prepared_parameters_and_independent_verification(tmp_path):
    handoff = _protocol(tmp_path)
    paths = preparation_paths(handoff)
    assert paths["command"] == handoff["jobs"][0]["command"][:-1]
    jobs = fit_jobs(paths, tmp_path / "continuation", python="python")
    assert [row["name"] for row in jobs] == ["native-fit", "native-fit-verify"]
    assert jobs[0]["command"] == paths["command"]
    assert "--plan-only" not in jobs[0]["command"]
    assert "--supervision-only" not in jobs[0]["command"]
    assert "--require-identifiable-supervision" in jobs[0]["command"]
    assert jobs[1]["command"].count("--bundle") == 2
    assert jobs[1]["command"][jobs[1]["command"].index("--directory") + 1] == str(tmp_path / "native")
    assert "--output" in jobs[1]["command"]
    assert all(row["max_invocations"] == 1 for row in jobs)


@pytest.mark.parametrize("fault", ["digest", "schema", "job_count", "different_parameters", "other_tool",
    "wrong_mode", "early_mode", "missing_strict", "different_cwd", "relative_cwd", "duplicate_output",
    "missing_bank", "missing_bundle", "duplicate_bundle", "empty_bundle", "nonstring_command"])
def test_an_altered_preparation_cannot_start_a_different_fit(tmp_path, fault):
    handoff = _protocol(tmp_path)
    left, right = (row["command"] for row in handoff["jobs"])
    if fault == "schema":
        handoff["schema"] = "other"
    elif fault == "job_count":
        handoff["jobs"].pop()
    elif fault == "different_parameters":
        right[right.index("--steps") + 1] = "606"
    elif fault == "wrong_mode":
        right[-1] = "--plan-only"
    elif fault == "different_cwd":
        handoff["jobs"][1]["cwd"] = str(tmp_path / "other")
    elif fault == "relative_cwd":
        for row in handoff["jobs"]:
            row["cwd"] = "relative"
    elif fault == "other_tool":
        left[1] = right[1] = "tools/other.py"
    elif fault == "nonstring_command":
        left[1] = right[1] = None
    elif fault == "early_mode":
        for command in (left, right):
            command.insert(2, "--plan-only")
    elif fault == "missing_strict":
        for command in (left, right):
            command.remove("--require-identifiable-supervision")
    elif fault in {"duplicate_output", "missing_bank", "missing_bundle", "duplicate_bundle", "empty_bundle"}:
        for command in (left, right):
            if fault == "duplicate_output":
                command[-1:-1] = ["--directory", "other"]
            elif fault == "missing_bank":
                command[command.index("--bank") + 1] = "--not-a-path"
            elif fault == "missing_bundle":
                while "--bundle" in command:
                    index = command.index("--bundle")
                    del command[index:index + 2]
            elif fault == "duplicate_bundle":
                command[-1:-1] = ["--bundle", command[command.index("--bundle") + 1]]
            else:
                command[command.index("--bundle") + 1] = "one="
    if fault != "digest":
        handoff = _receipt(handoff, digest=fit_identity)
    else:
        handoff["receipt_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        preparation_paths(handoff)


def test_relative_preparation_paths_keep_the_original_checkout_meaning(tmp_path):
    handoff = _protocol(tmp_path)
    for row in handoff["jobs"]:
        row["cwd"] = str(tmp_path)
        command = row["command"]
        command[command.index("--directory") + 1] = "native"
        command[command.index("--bundle") + 1] = "one=one"
    paths = preparation_paths(_receipt(handoff, digest=fit_identity))
    assert paths["native"] == tmp_path / "native"
    assert paths["command"][paths["command"].index("--directory") + 1] == str(tmp_path / "native")
    assert paths["bundles"][0] == "one=" + str(tmp_path / "one")


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    from tools import run_semantic_native_fit_handoff as module
    from tools.semantic_native_identifiability import IDENTIFIABILITY_CONTRACT, save_identifiability_preflight

    native = tmp_path / "native"
    plan = _receipt({"objective": "grammar_choices", "captured_fit_ids": ["fit"],
        "calibration_ids": ["calibration"], "held_ids": ["held"], "implementation": {},
        "grammar_identifiability_contract": copy.deepcopy(IDENTIFIABILITY_CONTRACT)}, "plan_sha256")
    from tests.test_semantic_native_identifiability import decision

    rows = decision("fit") + decision("calibration")
    supervision = _receipt({"plan_sha256": plan["plan_sha256"], "rows": rows})
    _save_if_absent(native / "plan.json", plan)
    _save_if_absent(native / "supervision.json", supervision)
    preflight = save_identifiability_preflight(native, plan, supervision)
    receipt = _receipt({"schema": "aura.semantic_source_native_preparation.v1",
        "plan_sha256": plan["plan_sha256"], "supervision_receipt_sha256": supervision["receipt_sha256"],
        "preflight_receipt_sha256": preflight["receipt_sha256"], "model_weights_loaded": False,
        "qualification_evidence": False, "serving_authority": False}, digest=fit_identity)
    _save_if_absent(tmp_path / "native-preparation.json", receipt)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    return tmp_path, {"native": native}, receipt


def test_preparation_is_replayed_from_actual_scoring_inputs_before_loading(prepared):
    directory, paths, receipt = prepared
    assert verify_preparation(directory, paths) == receipt


@pytest.mark.parametrize("fault", ["plan", "supervision", "preflight", "missing_preflight", "loaded",
    "qualified", "serving", "checkpoint", "report", "implementation"])
def test_self_signed_green_preparation_or_prior_training_cannot_pass(prepared, fault):
    import json

    directory, paths, receipt = prepared
    native = paths["native"]
    if fault in {"plan", "supervision", "preflight", "loaded", "qualified", "serving"}:
        field = {"plan": "plan_sha256", "supervision": "supervision_receipt_sha256",
                 "preflight": "preflight_receipt_sha256", "loaded": "model_weights_loaded",
                 "qualified": "qualification_evidence", "serving": "serving_authority"}[fault]
        receipt[field] = "f" * 64 if fault in {"plan", "supervision", "preflight"} else True
        path = directory / "native-preparation.json"
        path.unlink()
        _save_if_absent(path, _receipt(receipt, digest=fit_identity))
    elif fault == "missing_preflight":
        (native / "identifiability-preflight.json").unlink()
    elif fault in {"checkpoint", "report"}:
        (native / ("checkpoint-0.json" if fault == "checkpoint" else "report.json")).write_text("{}")
    else:
        plan_path = native / "plan.json"
        plan = json.loads(plan_path.read_bytes())
        plan["implementation"] = {"missing.py": "f" * 64}
        # Even a fully rebound scoring receipt cannot excuse changed code.
        plan = _receipt(plan, "plan_sha256")
        supervision_path = native / "supervision.json"
        supervision = json.loads(supervision_path.read_bytes())
        supervision["plan_sha256"] = plan["plan_sha256"]
        supervision = _receipt(supervision)
        for path, document in ((plan_path, plan), (supervision_path, supervision)):
            path.unlink()
            _save_if_absent(path, document)
        from tools.semantic_native_identifiability import save_identifiability_preflight
        (native / "identifiability-preflight.json").unlink()
        preflight = save_identifiability_preflight(native, plan, supervision)
        receipt.update(plan_sha256=plan["plan_sha256"],
                       supervision_receipt_sha256=supervision["receipt_sha256"],
                       preflight_receipt_sha256=preflight["receipt_sha256"])
        (directory / "native-preparation.json").unlink()
        _save_if_absent(directory / "native-preparation.json", _receipt(receipt, digest=fit_identity))
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_preparation(directory, paths)


def test_preparation_uses_its_declared_numerical_receipt_not_native_json_hash(tmp_path):
    path = tmp_path / "handoff.json"
    body = {"schema": "aura.semantic_source_native_preparation_plan.v1", "jobs": [{"rows": [1, 2]}]}
    valid = _receipt(copy.deepcopy(body), digest=fit_identity)
    _save_if_absent(path, valid)
    assert verified_preparation_document(path) == valid
    path.unlink()
    _save_if_absent(path, _receipt(body))
    with pytest.raises(ValueError, match="digest"):
        verified_preparation_document(path)


@pytest.mark.parametrize("fault", [None, "unfitted", "preflight", "fit_failed", "verify_failed",
                                  "upstream_failed", "launch_environment"])
def test_supervised_fit_sequence_stops_at_the_failed_prerequisite(prepared, monkeypatch, fault):
    import json
    from types import SimpleNamespace

    from core.runtime import detached_subprocess_broker as broker
    from tools import adjudicate_semantic_native_micro_stages as adjudication
    from tools import run_detached_step, run_semantic_native_fit_handoff as module
    from tools import run_semantic_native_micro_stages as stages

    preparation, paths, receipt = prepared
    protocol = _protocol(preparation)
    _save_if_absent(preparation / "handoff.json", protocol)
    supervisor = preparation / "supervisor"
    supervised = {"plan_sha256": "a" * 64, "command": ["python", "-u",
        "tools/run_semantic_source_handoff.py", "--directory", str(preparation),
        "--wait-bank-supervisor", "bank"]}
    _save_if_absent(supervisor / run_detached_step.PLAN_FILE, supervised)
    monkeypatch.setattr(run_detached_step, "_verify_plan", lambda *_args: None)
    monkeypatch.setattr(run_detached_step, "_status", lambda *_args: supervised)
    monkeypatch.setattr(broker, "broker_available", lambda: True)
    calls = []

    def wait(*_args, **_kwargs):
        if fault == "upstream_failed":
            raise ValueError("upstream failed")
        return {"receipt_sha256": "b" * 64}

    def invoke(command, **_kwargs):
        calls.append(command)
        failed = (fault == "fit_failed" and len(calls) == 1
                  or fault == "verify_failed" and len(calls) == 2)
        return SimpleNamespace(returncode=1 if failed else 0, status="failed" if failed else "passed",
            timed_out=False, containment_verified=True, receipt_sha256="c" * 64)

    monkeypatch.setattr(stages, "wait_for_fit", wait)
    monkeypatch.setattr(broker, "run_brokered_process", invoke)
    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    if fault == "launch_environment":
        monkeypatch.delenv("MLX_ENABLE_TF32")
        monkeypatch.setattr(stages, "wait_for_fit", lambda *_args, **_kwargs: pytest.fail(
            "invalid arithmetic must be caught before waiting for preparation"))
    native = paths["native"]
    plan = json.loads((native / "plan.json").read_bytes())
    checkpoint = {"receipt_sha256": "d" * 64, "step": 0 if fault == "unfitted" else 303}
    monkeypatch.setattr(adjudication, "verified_native_fit", lambda *_args: (
        plan, checkpoint, {"receipt_sha256": "e" * 64}))
    if fault == "preflight":
        (native / "identifiability-preflight.json").unlink()
    directory = preparation / "continuation"
    monkeypatch.setattr(module.sys, "argv", ["handoff", "--preparation-supervisor", str(supervisor),
        "--preparation-directory", str(preparation), "--directory", str(directory)])
    if fault in {None, "unfitted"}:
        assert module.main() == (2 if fault == "unfitted" else 0)
        assert len(calls) == 2
        completed = json.loads((directory / "fit-completion.json").read_bytes())
        assert completed["learned_checkpoint_selected"] == (fault is None)
        assert completed["preparation_receipt_sha256"] == receipt["receipt_sha256"]
        assert all(completed[key] is False for key in (
            "semantic_correctness_measured", "general_transfer_proven", "qualification_evidence",
            "serving_authority"))
    else:
        with pytest.raises((ValueError, FileNotFoundError)):
            module.main()
        assert len(calls) == {"fit_failed": 1, "verify_failed": 2,
                              "preflight": 0, "upstream_failed": 0, "launch_environment": 0}[fault]
        assert not (directory / "fit-completion.json").exists()
