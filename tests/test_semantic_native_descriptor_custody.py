"""Native isolation uses the existing authority key, never a replacement key."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.run_semantic_source_handoff import native_preparation_jobs
from tools.train_semantic_native_program import require_native_cortex_spec


def test_explicit_custody_is_used_without_writing_or_copying_key(monkeypatch, tmp_path):
    import core.brain.llm.model_registry as registry

    key = tmp_path / "existing.key"
    key.write_bytes(b"existing-fixture-not-a-real-key")
    monkeypatch.delenv("AURA_CORTEX_AUTHORITY_KEY_FILE", raising=False)
    seen = []
    def descriptor(**kwargs):
        import os

        seen.append((kwargs, os.environ["AURA_CORTEX_AUTHORITY_KEY_FILE"]))
        return SimpleNamespace(exact_identity=True)
    monkeypatch.setattr(registry, "get_active_cortex_spec", descriptor)
    assert require_native_cortex_spec(key).exact_identity
    assert seen == [({"force_refresh": True}, str(key))]
    assert key.read_bytes() == b"existing-fixture-not-a-real-key"
    assert tuple(tmp_path.iterdir()) == (key,)
    with pytest.raises(ValueError, match="supervised environment"):
        require_native_cortex_spec(tmp_path / "different.key")


def test_missing_signed_descriptor_fails_before_native_training(monkeypatch):
    import core.brain.llm.model_registry as registry

    monkeypatch.setattr(registry, "get_active_cortex_spec", lambda **kwargs: None)
    with pytest.raises(ValueError, match="exact signed resident descriptor"):
        require_native_cortex_spec()


def test_preparation_binds_same_key_path_in_both_child_commands(tmp_path):
    paths = {"candidate": tmp_path / "candidate.json", "report": tmp_path / "report.json",
             "bundles": ["family=" + str(tmp_path / "features")]}
    key = tmp_path / "key"
    jobs = native_preparation_jobs(paths, tmp_path / "bank", tmp_path / "native", tmp_path,
                                  python="python", authority_key_file=key)
    assert jobs[0]["command"][:-1] == jobs[1]["command"][:-1]
    for job in jobs:
        command = job["command"]
        assert Path(command[command.index("--authority-key-file") + 1]) == key
