"""Source continuation uses completed coefficients, never a green exit alone."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.learning.semantic_binary_fit_verification import verify_binary_fit_checkpoints
from core.learning.semantic_bounded_binary_fit import binary_fit_checkpoint_scope
from core.learning.semantic_fit_checkpoint import fit_identity, read_fit_archive, write_fit_archive
from core.learning.semantic_program_compositional_campaign import fit_compositional_source_campaign
from tests.test_compositional_source_training import source_bundles  # noqa: F401
from tests.test_semantic_program_shared_transducer import _grounding
from tools.run_semantic_source_handoff import (
    handoff_jobs,
    source_fit_paths,
    verify_source_fit_artifacts,
    native_preparation_jobs,
    source_bank_directory,
    verify_source_bank,
    verify_native_launch_environment,
)


@pytest.fixture
def fitted(source_bundles, tmp_path):
    with binary_fit_checkpoint_scope(tmp_path, "a" * 64):
        result = fit_compositional_source_campaign(source_bundles,
            input_grounding=_grounding(), source_order_inputs=True, binary_solver="blocked_lbfgs")
    return result.model, tmp_path


def test_every_archive_matches_the_serialized_model_coefficient(fitted):
    model, directory = fitted
    verification = verify_binary_fit_checkpoints(model, directory)
    assert len(verification["heads"]) == 11
    assert len({row["head"] for row in verification["heads"]}) == 11
    assert verification["all_archives_verified"] is True
    assert verification["all_coefficients_match"] is True
    assert all(verification[key] is False for key in (
        "objective_recomputed", "semantic_accuracy_measured", "qualification_evidence", "serving_authority"))
    assert verification["receipt_sha256"] == fit_identity({key: value for key, value in
        verification.items() if key != "receipt_sha256"})


@pytest.mark.parametrize("fault", ["missing", "byte_change", "unconverged", "coefficient", "iterations", "center"])
def test_even_rechecksummed_but_wrong_archive_cannot_pass(fitted, fault):
    model, directory = fitted
    receipt = copy.deepcopy(model.training_receipt["binary_head_fit_checkpoints"])
    row = receipt["records"][0]
    path = directory / (row["objective_identity"] + ".npz")
    if fault == "missing":
        path.unlink()
    elif fault == "byte_change":
        path.write_bytes(path.read_bytes() + b"altered")
    else:
        body, arrays = read_fit_archive(path, ("weight", "center"))
        if fault == "unconverged":
            body["status"] = "iterating"
        elif fault == "coefficient":
            arrays["weight"][0] += .125
            row["final_parameters_sha256"] = fit_identity(arrays["weight"])
        elif fault == "iterations":
            body["iterations"] += 1
        else:
            arrays["center"][0] += 1
        write_fit_archive(path, body, arrays)
        import hashlib
        receipt["final_archives"][row["objective_identity"]] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt["receipt_sha256"] = fit_identity({key: value for key, value in receipt.items()
                                               if key != "receipt_sha256"})
    altered = SimpleNamespace(**{name: getattr(model, name) for name in (
        "operation_pointer", "argument_pointer", "definition_pointer", "argument_role_heads",
        "argument_proposal_heads", "definition_relation_head", "receipt_sha256")},
        training_receipt={**model.training_receipt, "binary_head_fit_checkpoints": receipt})
    with pytest.raises((ValueError, FileNotFoundError)):
        verify_binary_fit_checkpoints(altered, directory)


def test_partial_convergence_inventory_cannot_be_mistaken_for_all_heads(fitted):
    model, directory = fitted
    receipt = copy.deepcopy(model.training_receipt["binary_head_fit_checkpoints"])
    receipt["records"].pop()
    receipt["receipt_sha256"] = fit_identity({key: value for key, value in receipt.items()
                                               if key != "receipt_sha256"})
    model.training_receipt["binary_head_fit_checkpoints"] = receipt
    with pytest.raises(ValueError, match="inventory"):
        verify_binary_fit_checkpoints(model, directory)


def _plan():
    return {"cwd": "/tmp/frozen-source", "command": ["python", "-u",
        "tools/train_compositional_semantic_sources.py", "--bundle", "one=/tmp/source-one",
        "--bundle", "two=/tmp/source-two", "--source-order-inputs", "--binary-solver", "blocked_lbfgs",
        "--output", "/tmp/fit/candidate.json", "--report-output", "/tmp/fit/report.json",
        "--binary-checkpoints", "/tmp/fit/checkpoints"]}


def test_handoff_freezes_exact_measured_source_paths_and_disjoint_training(tmp_path):
    paths = source_fit_paths(_plan())
    jobs = handoff_jobs(paths, tmp_path, python="python")
    assert [item["name"] for item in jobs] == ["freeze-source-folds", "fit-source-bank"]
    freeze, bank = (item["command"] for item in jobs)
    assert freeze[freeze.index("--transducer") + 1] == str(Path("/tmp/fit/candidate.json").resolve())
    assert "--source-fit-only" in freeze and freeze[freeze.index("--axis") + 1] == "utterance"
    assert bank[bank.index("--folds") + 1] == str(tmp_path / "utterance-folds.json")
    assert "--reuse-candidate" not in bank and "--held-source-id" not in bank
    assert bank[bank.index("--binary-checkpoints") + 1] == str(tmp_path / "bank/binary-checkpoints")
    assert bank[bank.index("--per-construction") + 1] == "1"
    assert bank.count("--bundle") == freeze.count("--bundle") == 2


@pytest.mark.parametrize("fault", ["other_tool", "ordinary_solver", "duplicate_bundle", "duplicate_output", "missing_value"])
def test_handoff_cannot_change_or_guess_the_supervised_target(fault):
    plan = _plan()
    command = plan["command"]
    if fault == "other_tool":
        command[2] = "tools/other.py"
    elif fault == "ordinary_solver":
        command[command.index("--binary-solver") + 1] = "liblinear"
    elif fault == "duplicate_bundle":
        command.extend(["--bundle", "one=/tmp/other"])
    elif fault == "duplicate_output":
        command.extend(["--output", "/tmp/other.json"])
    else:
        command.append("--report-output")
    with pytest.raises(ValueError):
        source_fit_paths(plan)


def test_source_artifact_handoff_binds_report_and_actual_coefficients(source_bundles, tmp_path):
    from core.learning.semantic_program_campaign import _sha
    from tools.probe_semantic_proposer_crossfit import _save_if_absent

    checkpoints = tmp_path / "checkpoints"
    with binary_fit_checkpoint_scope(checkpoints, "a" * 64):
        result = fit_compositional_source_campaign(source_bundles,
            input_grounding=_grounding(), source_order_inputs=True, binary_solver="blocked_lbfgs")
    paths = {"candidate": tmp_path / "candidate.json", "report": tmp_path / "report.json",
             "checkpoints": checkpoints}
    _save_if_absent(paths["candidate"], result.model.to_dict())
    _save_if_absent(paths["report"], result.report)
    verification = verify_source_fit_artifacts(paths)
    assert verification["source_fit_complete"] is True
    assert verification["validation_ids_sha256"] == result.report["validation_example_ids_sha256"]
    assert verification["qualification_evidence"] is False
    assert len(verification["binary_verification"]["heads"]) == 11
    body = {key: value for key, value in result.report.items() if key != "report_sha256"}
    body["fit_complete"] = False
    paths["report"].unlink()
    _save_if_absent(paths["report"], {**body, "report_sha256": _sha(body)})
    with pytest.raises(ValueError, match="incomplete"):
        verify_source_fit_artifacts(paths)


def test_native_preparation_keeps_one_protocol_without_loading_or_repeating_a_fit(tmp_path):
    paths = source_fit_paths(_plan())
    jobs = native_preparation_jobs(paths, tmp_path / "bank", tmp_path / "native", tmp_path,
                                   python="python")
    assert [item["name"] for item in jobs] == ["native-plan", "native-supervision"]
    left, right = (item["command"] for item in jobs)
    assert left[:-1] == right[:-1]
    assert left[-1] == "--plan-only" and right[-1] == "--supervision-only"
    assert "--require-identifiable-supervision" in left
    for flag, value in {"--steps": "303", "--save-every": "101", "--joint-graph-contrasts": "4",
                        "--register-encoding": "role_relative_v1",
                        "--source-pair-policy": "typed_choice_complete_v1",
                        "--schedule-policy": "construction_depth_balanced_v1"}.items():
        assert left[left.index(flag) + 1] == value
    assert left.count("--bundle") == len(paths["bundles"])
    assert "--reuse-prefix-from" not in left


@pytest.mark.parametrize("value", [None, "1", "false"])
def test_native_launch_refuses_wrong_arithmetic_without_repairing_environment(monkeypatch, value):
    import os

    if value is None:
        monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    else:
        monkeypatch.setenv("MLX_ENABLE_TF32", value)
    with pytest.raises(ValueError, match="process launch"):
        verify_native_launch_environment(["python", "train.py", "--precision", "float32",
                                          "--prefix-strategy", "trie"])
    assert os.environ.get("MLX_ENABLE_TF32") == value


def test_native_launch_uses_the_same_cpu_only_execution_contract(monkeypatch):
    from tools.semantic_native_execution import execution_contract

    monkeypatch.setenv("MLX_ENABLE_TF32", "0")
    assert verify_native_launch_environment(["python", "train.py", "--precision", "float32",
        "--prefix-strategy", "trie"]) == execution_contract(precision="float32", prefix_strategy="trie")


@pytest.mark.parametrize("options", [[], ["--precision", "float32"],
    ["--precision", "float32", "--precision", "float32", "--prefix-strategy", "trie"],
    ["--precision", "--prefix-strategy", "trie"]])
def test_native_launch_does_not_guess_missing_or_repeated_arithmetic(options):
    with pytest.raises(ValueError, match="explicit arithmetic"):
        verify_native_launch_environment(["python", "train.py", *options])


@pytest.mark.parametrize("policy_only", [False, True])
def test_preparation_rejects_bad_launch_before_waiting_or_freezing_jobs(tmp_path, monkeypatch, policy_only):
    from tools import run_detached_step, run_semantic_native_micro_stages, run_semantic_source_handoff
    from tools.probe_semantic_proposer_crossfit import _save_if_absent

    source = tmp_path / "source-supervisor"
    bank = tmp_path / "bank-supervisor"
    _save_if_absent(source / run_detached_step.PLAN_FILE, _plan())
    _save_if_absent(bank / run_detached_step.PLAN_FILE, {"cwd": str(tmp_path), "command": [
        "python", "-u", "tools/run_semantic_source_handoff.py", "--source-fit-supervisor",
        str(source), "--directory", str(tmp_path / "bank")]})
    monkeypatch.setattr(run_detached_step, "_verify_plan", lambda *_args: None)
    monkeypatch.setattr(run_semantic_native_micro_stages, "wait_for_fit",
        lambda *_args, **_kwargs: pytest.fail("bad arithmetic cannot wait for a fit"))
    monkeypatch.delenv("MLX_ENABLE_TF32", raising=False)
    directory = tmp_path / "preparation"
    command = ["handoff", "--source-fit-supervisor", str(source), "--wait-bank-supervisor",
        str(bank), "--native-directory", str(tmp_path / "native"), "--directory", str(directory)]
    if policy_only:
        command += ["--policy-output", str(directory / "broker-policy.json")]
    monkeypatch.setattr(run_semantic_source_handoff.sys, "argv", command)
    with pytest.raises(ValueError, match="process launch"):
        run_semantic_source_handoff.main()
    assert not (directory / "handoff.json").exists()
    assert not (directory / "broker-policy.json").exists()


def test_source_bank_supervisor_binds_the_original_fit_and_output(tmp_path):
    supervisor = tmp_path / "fit-supervisor"
    plan = {"cwd": str(tmp_path), "command": ["python", "-u", "tools/run_semantic_source_handoff.py",
        "--source-fit-supervisor", str(supervisor), "--directory", "bank"]}
    assert source_bank_directory(plan, supervisor) == tmp_path / "bank"
    with pytest.raises(ValueError, match="another"):
        source_bank_directory(plan, tmp_path / "other")
    plan["command"].extend(["--native-directory", "other"])
    with pytest.raises(ValueError, match="supervised"):
        source_bank_directory(plan, supervisor)


@pytest.mark.parametrize("fault", [None, "candidate", "report", "folds", "missing_row", "stale_parent"])
def test_completed_bank_cannot_change_the_source_basis(tmp_path, fault):
    import hashlib
    from tools.probe_semantic_proposer_crossfit import _digest, _save_if_absent

    candidate, source, folds = (tmp_path / name for name in ("candidate.json", "source.json", "utterance-folds.json"))
    for path in (candidate, source, folds):
        path.write_bytes(b"{}")
    sha = hashlib.sha256(b"{}").hexdigest()
    verification = {"candidate_receipt_sha256": "a" * 64, "candidate_sha256": sha,
                    "source_report_sha256": sha}
    plan = {"schema": "aura.semantic_proposer_crossfit_plan.v1", "held_ids": ["held"],
            "parent_receipt_sha256": "a" * 64, "source_report_sha256": sha, "folds_sha256": sha}
    if fault == "stale_parent":
        plan["parent_receipt_sha256"] = "b" * 64
    plan["plan_sha256"] = _digest(plan)
    report = {"schema": "aura.semantic_proposer_crossfit.v1", "plan_sha256": plan["plan_sha256"],
              "row_receipts": {} if fault == "missing_row" else {"held": "c" * 64}}
    report["receipt_sha256"] = _digest(report)
    _save_if_absent(tmp_path / "bank/plan.json", plan)
    _save_if_absent(tmp_path / "bank/report.json", report)
    if fault in {"candidate", "report", "folds"}:
        {"candidate": candidate, "report": source, "folds": folds}[fault].write_bytes(b"{\"altered\":true}")
    paths = {"candidate": candidate, "report": source}
    if fault is None:
        assert verify_source_bank(tmp_path, paths, verification) == report
    else:
        with pytest.raises(ValueError):
            verify_source_bank(tmp_path, paths, verification)
