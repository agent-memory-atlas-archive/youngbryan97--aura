"""A fresh optimizer may read old states, but cannot inherit old evidence."""

import hashlib
import json
from copy import deepcopy

import mlx.core as mx
import pytest

from core.learning.frozen_state_store import FrozenStateStore
from tools.evaluate_semantic_native_checkpoint import digest
from tools.semantic_native_prefix_reuse import open_reused_prefix, prefix_reuse_contract
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


@pytest.mark.parametrize("defect", ["population", "model", "other_implementation", "source_rows",
                                    "missing_manifest", "manifest_digest"])
def test_reuse_refuses_protocol_and_manifest_drift(tmp_path, defect):
    source, new, _row, _manifest, _supervision = fixture(tmp_path)
    if defect == "population":
        new["suffix_layers"] = 2
    elif defect == "model":
        new["model_descriptor_sha256"] = "other"
    elif defect == "other_implementation":
        new["implementation"]["core/learning/semantic_native_grammar.py"] = "different"
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
