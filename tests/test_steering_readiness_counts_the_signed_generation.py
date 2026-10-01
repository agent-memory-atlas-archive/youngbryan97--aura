"""Steering readiness reports the tissue the engine attaches.

On 30 September the integrity audit said "CAA steering at 30.0% (bootstrap)"
for a cortex whose migration contract qualified a signed generation of 80
vectors, which the steering engine materializes from custody and hooks at the
layers that generation carries. The report read only training/vectors, where
every vector belongs to an earlier model, and expected the layers of a depth
band the engine does not use.
"""
from __future__ import annotations

import pytest

from core.consciousness.caa import readiness_report
from tests.support.cortex_migration_authority import build_signed_migration_authorities

KEYS = ("valence_positive", "arousal", "curiosity", "frustration", "energy")
LAYERS = (3, 7)
DIGEST = "a" * 64


@pytest.fixture()
def signed(tmp_path, monkeypatch):
    names = tuple(f"{key}_layer{layer}.npz" for key in KEYS for layer in LAYERS)
    state = tmp_path / "state"
    authorities = build_signed_migration_authorities(
        tmp_path / "fixture", descriptor_sha256=DIGEST, state_root=state, vector_names=names
    )
    monkeypatch.setenv(
        "AURA_CORTEX_AUTHORITY_KEY_FILE", str(state / "private/cortex-upgrade/migration-authority.key")
    )
    active = {
        "path": "/models/test-cortex",
        "fused_at": 1.0,
        "descriptor_sha256": DIGEST,
        "artifact_profile": {"num_hidden_layers": 64},
        "identity_valid": True,
        "identity_error": "",
        "steering_authority_status": "qualified",
        "steering_authority_kind": "caa_model_bound",
        "steering_authority": authorities["steering"],
    }
    monkeypatch.setattr(readiness_report, "_active_model", lambda _dir: active)
    return authorities["steering"]


def test_a_signed_generation_is_counted_at_the_layers_it_carries(signed, tmp_path) -> None:
    report = readiness_report.verify_readiness(
        vectors_dir=tmp_path / "no-vectors-here", fused_model_dir=tmp_path
    )
    assert report["runtime_contract"]["expected_layers"] == list(LAYERS)
    assert report["runtime_contract"]["expected_extracted"] == len(KEYS) * len(LAYERS)
    assert report["level"] == "production"
    assert report["steering_capacity_pct"] == 100.0


def test_a_vector_changed_after_signing_counts_for_nothing(signed, tmp_path) -> None:
    from pathlib import Path

    tampered = Path(signed["evidence"]["vector:arousal_layer3.npz"]["path"])
    tampered.write_bytes(b"not what was signed")
    report = readiness_report.verify_readiness(
        vectors_dir=tmp_path / "no-vectors-here", fused_model_dir=tmp_path
    )
    assert report["level"] != "production"
    assert report["runtime_contract"]["expected_extracted"] == 0
