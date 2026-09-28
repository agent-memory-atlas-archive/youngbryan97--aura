"""A fresh optimizer may read old states, but cannot inherit old evidence."""

import hashlib
import json
from copy import deepcopy

import mlx.core as mx
import pytest

from core.learning.frozen_state_store import FrozenStateStore
from core.learning.semantic_native_decision_supervision import GRAMMAR_CHOICE_CONTRACT
from core.learning.semantic_native_source_pairs import SOURCE_PAIR_CONTRACT
from core.learning.semantic_native_path_objective import (
    GRAMMAR_PATH_CONTRACT,
    JOINT_GRAPH_CONTRAST_CONTRACT,
    path_choice_contract,
)
from core.learning.semantic_native_path_selection import JOINT_GRAPH_SELECTION_CONTRACT
from tools.evaluate_semantic_native_checkpoint import digest
from tools.semantic_native_prefix_reuse import open_reused_prefix, prefix_reuse_contract
from tools.train_semantic_native_program import graft_reused_grammar_states
from tools.verify_semantic_native_fit import verify_state_storage


def fixture(tmp_path):
    source = tmp_path / "old"
    states = FrozenStateStore(source / "prefix-states", plan_sha256="a" * 64,
                              max_resident_bytes=64)
    row = {"source": "source", "decision_index": 0, "choice_index": 0,
           "tokens": [1, 2, 3], "semantic_positions": [2],
           "continuation_start": 2}
    key = ("source", 0, 0)
    with mx.stream(mx.cpu):
        states.write_source("source", {key: mx.arange(4, dtype=mx.float32)},
                            sequence_digests={key: digest(row["tokens"])})
    prior = {"schema": "aura.semantic_native_fit_plan.v3", "plan_sha256": "a" * 64,
             "prefix_storage_contract": {"schema": "aura.frozen_state_storage_contract.v1",
                 "mode": "source_shards", "max_resident_bytes": 64,
                 "lossy_compression": False, "all_alternatives_retained": True},
             "execution_contract": {"prefix_strategy": "trie"},
             "model_descriptor_sha256": "model", "suffix_layers": 1,
             "implementation": {"tools/train_semantic_native_program.py": "old",
                                "core/learning/frozen_state_store.py": "old",
                                "core/learning/semantic_native_grammar.py": "same"}}
    prior.pop("plan_sha256")
    prior["plan_sha256"] = digest(prior)
    # Rebind the tiny shard fixture to its actual frozen plan identity.
    old = source / "prefix-states" / (hashlib.sha256(b"source").hexdigest() + ".json")
    manifest = json.loads(old.read_text())
    manifest["plan_sha256"] = prior["plan_sha256"]
    manifest.pop("receipt_sha256")
    manifest["receipt_sha256"] = digest(manifest)
    old.chmod(0o600)
    old.write_text(json.dumps(manifest, sort_keys=True))
    (source / "plan.json").write_text(json.dumps(prior))
    supervision = {"plan_sha256": prior["plan_sha256"], "rows": [row]}
    supervision["receipt_sha256"] = digest(supervision)
    (source / "supervision.json").write_text(json.dumps(supervision))
    (source / "prefix-receipts").mkdir()
    (source / "prefix-receipts" / "source.json").write_text(json.dumps({
        "source": "source", "plan_sha256": prior["plan_sha256"], "checked_choices": 1}))
    new = deepcopy(prior)
    new["implementation"].update({"tools/train_semantic_native_program.py": "new",
                                  "core/learning/frozen_state_store.py": "new",
                                  "tools/semantic_native_prefix_reuse.py": "new"})
    return source, new, row, manifest, supervision


def test_full_reuse_contract_reopens_all_states_and_verifies_report(tmp_path):
    source, new, row, manifest, _old_supervision = fixture(tmp_path)
    contract = prefix_reuse_contract(source, new)
    new["reused_prefix_contract"] = contract
    new.pop("plan_sha256")
    new["plan_sha256"] = digest(new)
    supervision = {"plan_sha256": new["plan_sha256"], "rows": [row]}
    with mx.stream(mx.cpu):
        states = open_reused_prefix(contract, new, supervision)
        assert mx.array_equal(states[("source", 0, 0)], mx.arange(4, dtype=mx.float32)).item()
    report = {"reused_prefix_contract": contract, "prefix_storage_receipt": states.receipt(),
              "prefix_capture_receipts": [{"source": "source", "checked_choices": 1}]}
    assert verify_state_storage(tmp_path / "new", new, report, supervision)["sequences"] == 1
    report["prefix_capture_receipts"][0]["checked_choices"] = 0
    with pytest.raises(ValueError, match="capture inventory differs"):
        verify_state_storage(tmp_path / "new", new, report, supervision)
    assert manifest["plan_sha256"] == contract["source_plan_sha256"]


def test_joint_graph_reuse_grafts_old_grammar_and_new_graph_into_verified_shard(tmp_path):
    source, new, row, _manifest, _old_supervision = fixture(tmp_path)
    prior = json.loads((source / "plan.json").read_text())
    prior.update(objective="grammar_choices", loss_scope="semantic_decisions",
                 grammar_choice_contract=GRAMMAR_CHOICE_CONTRACT,
                 unfitted_checkpoint_eligible=True,
                 selection="minimum_source_calibration_conditional_grammar_choice_loss")
    prior.pop("plan_sha256")
    prior["plan_sha256"] = digest(prior)
    (source / "plan.json").write_text(json.dumps(prior))
    manifest_path = next((source / "prefix-states").glob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["plan_sha256"] = prior["plan_sha256"]
    manifest.pop("receipt_sha256")
    manifest["receipt_sha256"] = digest(manifest)
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps(manifest))
    prior_supervision = {"plan_sha256": prior["plan_sha256"], "rows": [row]}
    prior_supervision["receipt_sha256"] = digest(prior_supervision)
    (source / "supervision.json").write_text(json.dumps(prior_supervision))
    capture_path = source / "prefix-receipts" / "source.json"
    capture = json.loads(capture_path.read_text())
    capture["plan_sha256"] = prior["plan_sha256"]
    capture_path.write_text(json.dumps(capture))

    new = deepcopy(prior)
    new.update(schema="aura.semantic_native_fit_plan.v7",
               grammar_choice_contract=path_choice_contract(),
               grammar_path_objective_contract=GRAMMAR_PATH_CONTRACT,
               path_checkpoint_selection_contract=JOINT_GRAPH_SELECTION_CONTRACT,
               selection="baseline_preserving_joint_source_calibration",
               joint_graph_contrast_limit=2,
               graph_contrast_contract=JOINT_GRAPH_CONTRAST_CONTRACT)
    new["implementation"].update({"tools/train_semantic_native_program.py": "new",
                                  "core/learning/frozen_state_store.py": "new",
                                  "core/learning/semantic_native_grammar.py": "new"})
    contract = prefix_reuse_contract(source, new)
    assert contract["schema"] == "aura.native_frozen_prefix_reuse.v2"
    assert contract["row_bound_implementation_paths"] == ["core/learning/semantic_native_grammar.py"]
    new["reused_prefix_contract"] = contract
    new.pop("plan_sha256")
    new["plan_sha256"] = digest(new)
    graph_rows = [{"source": "source", "choice_index": index, "tokens": [1, 2, 4 + index]}
                  for index in range(2)]
    supervision = {"plan_sha256": new["plan_sha256"], "rows": [row],
                   "graph_rows": graph_rows}
    with mx.stream(mx.cpu):
        prior_states = open_reused_prefix(contract, new, supervision)
        old_key, graph_key, rival_key = ("source", 0, 0), ("source", -1, 0), ("source", -1, 1)
        combined = graft_reused_grammar_states(
            prior_states, "source", (((old_key,), 0),),
            {graph_key: mx.arange(4, dtype=mx.float32) + 10,
             rival_key: mx.arange(4, dtype=mx.float32) + 20})
        store = FrozenStateStore(tmp_path / "new" / "prefix-states",
            plan_sha256=new["plan_sha256"], max_resident_bytes=64)
        store.write_source("source", combined, sequence_digests={
            old_key: digest(row["tokens"]),
            graph_key: digest(graph_rows[0]["tokens"]),
            rival_key: digest(graph_rows[1]["tokens"])})
    captures = {"plan_sha256": new["plan_sha256"], "source": "source",
                "checked_choices": 2, "complete_rankings_checked": True}
    (tmp_path / "new" / "prefix-receipts").mkdir()
    (tmp_path / "new" / "prefix-receipts" / "source.json").write_text(json.dumps(captures))
    report = {"reused_prefix_contract": contract, "prefix_storage_receipt": store.receipt(),
              "prefix_capture_receipts": [{key: value for key, value in captures.items()
                                           if key != "plan_sha256"}]}
    result = verify_state_storage(tmp_path / "new", new, report, supervision)
    assert result["sequences"] == 3
    report["prefix_capture_receipts"][0]["checked_choices"] = 1
    with pytest.raises(ValueError, match="capture inventory differs"):
        verify_state_storage(tmp_path / "new", new, report, supervision)
    report["prefix_capture_receipts"][0]["checked_choices"] = 2
    old_shard = next((source / "prefix-states").glob("*.safetensors"))
    old_shard.chmod(0o600)
    with old_shard.open("r+b") as handle:
        handle.seek(-1, 2)
        last = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes((last[0] ^ 1,)))
    with pytest.raises(ValueError, match="reused frozen source bytes"):
        verify_state_storage(tmp_path / "new", new, report, supervision)


def test_reuse_accepts_unchanged_implementation_and_rejects_capture_receipt_drift(tmp_path):
    source, new, _row, _manifest, _supervision = fixture(tmp_path)
    prior = json.loads((source / "plan.json").read_text())
    new["implementation"] = prior["implementation"]
    contract = prefix_reuse_contract(source, new)
    assert contract["changed_implementation_paths"] == []
    capture = source / "prefix-receipts" / "source.json"
    body = json.loads(capture.read_text())
    body["checked_choices"] = 0
    capture.write_text(json.dumps(body))
    assert prefix_reuse_contract(source, new)["capture_inventory_sha256"] != contract["capture_inventory_sha256"]


@pytest.mark.parametrize("path_mode", [False, True, "typed"])
def test_reuse_admits_only_bound_paired_objective_on_identical_frozen_sequences(tmp_path, path_mode):
    source, new, row, _manifest, _supervision = fixture(tmp_path)
    prior = json.loads((source / "plan.json").read_text())
    prior.update(objective="grammar_choices", loss_scope="semantic_decisions",
                 grammar_choice_contract=GRAMMAR_CHOICE_CONTRACT,
                 unfitted_checkpoint_eligible=True,
                 selection="minimum_source_calibration_conditional_grammar_choice_loss")
    prior.pop("plan_sha256")
    prior["plan_sha256"] = digest(prior)
    (source / "plan.json").write_text(json.dumps(prior))
    manifest_path = next((source / "prefix-states").glob("*.json"))
    manifest = json.loads(manifest_path.read_text())
    manifest["plan_sha256"] = prior["plan_sha256"]
    manifest.pop("receipt_sha256")
    manifest["receipt_sha256"] = digest(manifest)
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps(manifest))
    supervision = {"plan_sha256": prior["plan_sha256"], "rows": [row]}
    supervision["receipt_sha256"] = digest(supervision)
    (source / "supervision.json").write_text(json.dumps(supervision))
    receipt = source / "prefix-receipts" / "source.json"
    body = json.loads(receipt.read_text())
    body["plan_sha256"] = prior["plan_sha256"]
    receipt.write_text(json.dumps(body))
    new = deepcopy(prior)
    new.update(schema="aura.semantic_native_fit_plan.v4", objective="grammar_source_pairs",
               grammar_source_pair_contract=SOURCE_PAIR_CONTRACT,
               grammar_source_pair_fit_partners={"source": {"partner": "peer"}},
               grammar_source_pair_updates=1)
    new["implementation"].update({"tools/train_semantic_native_program.py": "new",
                                   "core/learning/semantic_native_source_pairs.py": "new"})
    if path_mode:
        from core.learning.semantic_native_path_objective import (
            GRAMMAR_PATH_CONTRACT,
            path_choice_contract,
        )
        from core.learning.semantic_native_path_selection import PATH_SELECTION_CONTRACT

        new.update(schema="aura.semantic_native_fit_plan.v5",
                   grammar_choice_contract=path_choice_contract(),
                   grammar_path_objective_contract=GRAMMAR_PATH_CONTRACT,
                   path_checkpoint_selection_contract=PATH_SELECTION_CONTRACT,
                   selection="baseline_preserving_complete_source_calibration_paths")
        new["implementation"]["core/learning/semantic_native_path_objective.py"] = "new"
        if path_mode == "typed":
            from tests.test_semantic_native_source_control import typed_plan

            new.update(typed_plan())
            new["implementation"]["core/learning/semantic_native_typed_source_pairs.py"] = "new"
            # This fixture changes only training supervision, not captured source inventory.
            prior.update(fit_ids=["a", "b"], scheduled_fit_ids=["a", "b"])
            prior.pop("plan_sha256")
            prior["plan_sha256"] = digest(prior)
            (source / "plan.json").write_text(json.dumps(prior))
            manifest["plan_sha256"] = prior["plan_sha256"]
            manifest.pop("receipt_sha256")
            manifest["receipt_sha256"] = digest(manifest)
            manifest_path.write_text(json.dumps(manifest))
            supervision["plan_sha256"] = prior["plan_sha256"]
            supervision.pop("receipt_sha256")
            supervision["receipt_sha256"] = digest(supervision)
            (source / "supervision.json").write_text(json.dumps(supervision))
            body["plan_sha256"] = prior["plan_sha256"]
            receipt.write_text(json.dumps(body))
    contract = prefix_reuse_contract(source, new)
    assert contract["source_count"] == 1
    assert contract["optimizer_state_reused"] is False
    new["suffix_layers"] = 2
    with pytest.raises(ValueError, match="differs"):
        prefix_reuse_contract(source, new)
    new["suffix_layers"] = 1
    new["input"] = "source_erased"
    with pytest.raises(ValueError, match="undeclared"):
        prefix_reuse_contract(source, new)
    new.pop("input")
    if path_mode:
        new["path_checkpoint_selection_contract"] = {}
        with pytest.raises(ValueError, match="contract differs"):
            prefix_reuse_contract(source, new)


@pytest.mark.parametrize("defect", ["population", "model", "other_implementation", "path_relabel", "typed_relabel", "source_rows",
                                    "missing_manifest", "manifest_digest"])
def test_reuse_refuses_protocol_and_manifest_drift(tmp_path, defect):
    source, new, _row, _manifest, _supervision = fixture(tmp_path)
    if defect == "population":
        new["suffix_layers"] = 2
    elif defect == "model":
        new["model_descriptor_sha256"] = "other"
    elif defect == "other_implementation":
        new["implementation"]["core/learning/semantic_native_grammar.py"] = "different"
    elif defect == "path_relabel":
        new["implementation"]["core/learning/semantic_native_path_objective.py"] = "new"
    elif defect == "typed_relabel":
        new["implementation"]["core/learning/semantic_native_typed_source_pairs.py"] = "new"
    elif defect == "source_rows":
        path = source / "supervision.json"
        body = json.loads(path.read_text())
        body["rows"][0]["tokens"] = [9, 9, 9]
        body.pop("receipt_sha256")
        body["receipt_sha256"] = digest(body)
        path.write_text(json.dumps(body))
    elif defect == "missing_manifest":
        next((source / "prefix-states").glob("*.json")).unlink()
    else:
        path = next((source / "prefix-states").glob("*.json"))
        body = json.loads(path.read_text())
        body["array_bytes"] += 1
        path.write_text(json.dumps(body))
    if defect == "source_rows":
        # The contract can bind an internally different, still consistent
        # supervision only until its shard sequence digests are checked.
        with pytest.raises(ValueError):
            prefix_reuse_contract(source, new)
    else:
        with pytest.raises(ValueError):
            prefix_reuse_contract(source, new)
