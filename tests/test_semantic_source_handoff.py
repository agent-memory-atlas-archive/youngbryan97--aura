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
