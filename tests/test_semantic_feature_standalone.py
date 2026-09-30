"""A contained acquisition reuses the resident algorithm and never fabricates ownership."""

import asyncio
import os
from types import SimpleNamespace

import numpy as np
import pytest

from tools.semantic_feature_standalone import StandaloneSemanticFeatureClient, wait_for_lane_release


def status(*, terminal=False):
    return {"plan_sha256": "a" * 64, "terminal": terminal, "supervisor_alive": not terminal,
            "completion_indeterminate": False, "child_state": "dead" if terminal else "alive",
            "state": "running", "receipt": {"plan_sha256": "a" * 64,
            "containment_verified": True, "passed": False, "returncode": 2,
            "receipt_sha256": "b" * 64}}


def test_failed_experiment_can_release_the_lane_only_after_authoritative_cleanup():
    states, sleeps = [status(), status(terminal=True)], []
    receipt = wait_for_lane_release("run", plan_sha256="a" * 64, timeout=60,
        inspect=lambda _path: states.pop(0), clock=lambda: 0., sleep=sleeps.append)
    assert receipt["passed"] is False and sleeps == [30.]


@pytest.mark.parametrize("defect", ["plan", "receipt_plan", "containment", "child", "supervisor", "indeterminate"])
def test_unknown_or_changed_predecessor_never_becomes_permission_to_load(defect):
    measured = status(terminal=defect not in {"supervisor", "indeterminate"})
    if defect == "plan":
        measured["plan_sha256"] = "c" * 64
    elif defect == "receipt_plan":
        measured["receipt"]["plan_sha256"] = "c" * 64
    elif defect == "containment":
        measured["receipt"]["containment_verified"] = False
    elif defect == "child":
        measured["child_state"] = "alive"
    elif defect == "supervisor":
        measured["supervisor_alive"] = False
    else:
        measured["completion_indeterminate"] = True
    with pytest.raises(ValueError):
        wait_for_lane_release("run", plan_sha256="a" * 64, timeout=60,
            inspect=lambda _path: measured, sleep=lambda _seconds: pytest.fail("unsafe wait"))


@pytest.mark.parametrize("bound", [0, -1, float("inf"), float("nan"), True, 86401])
def test_invalid_wait_bounds_refuse_before_inspection(bound):
    with pytest.raises(ValueError, match="finite bound"):
        wait_for_lane_release("run", plan_sha256="a" * 64, timeout=bound,
            inspect=lambda _path: pytest.fail("invalid wait inspected predecessor"))


def test_wait_exhaustion_is_explicit_and_bounded():
    now, sleeps = [0.], []
    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += seconds
    with pytest.raises(ValueError, match="no model was loaded"):
        wait_for_lane_release("run", plan_sha256="a" * 64, timeout=31,
            inspect=lambda _path: status(), clock=lambda: now[0], sleep=sleep)
    assert sleeps == [30., 1.]


@pytest.fixture
def client(tmp_path):
    owner = {"owner_id": "standalone:test", "exclusive": True, "fencing_token": 3,
             "process": {"pid": os.getpid()}, "model_path": str(tmp_path)}
    controller = SimpleNamespace(heartbeat_owner_sync=lambda *_args, **_kwargs: True,
                                 snapshot=lambda: {"owners": [owner]})
    decision = SimpleNamespace(owner_id=owner["owner_id"], fencing_token=3,
        receipt_id="committed", admitted=True, ready_to_spawn=True,
        state=SimpleNamespace(value="committed"))
    lease = SimpleNamespace(active=True, inherited=False, controller=controller, decision=decision)
    envelope = SimpleNamespace(reclaim=lambda _step: None)
    result = StandaloneSemanticFeatureClient(tmp_path, lease=lease, envelope=envelope)
    result.test_owner = owner
    return result


@pytest.mark.parametrize("defect", ["inactive", "inherited", "uncommitted", "lost", "pid", "fence", "shared", "other"])
def test_owned_process_and_exclusive_fence_are_checked_before_any_load(client, monkeypatch, defect):
    import mlx_lm.utils
    monkeypatch.setattr(mlx_lm.utils, "load", lambda *_args: pytest.fail("unowned model load"))
    if defect == "inactive":
        client.lease.active = False
    elif defect == "inherited":
        client.lease.inherited = True
    elif defect == "uncommitted":
        client.lease.decision.state.value = "reserved"
    elif defect == "lost":
        client.lease.controller.heartbeat_owner_sync = lambda *_args, **_kwargs: False
    elif defect == "pid":
        client.test_owner["process"]["pid"] += 1
    elif defect == "fence":
        client.test_owner["fencing_token"] += 1
    elif defect == "shared":
        client.test_owner["exclusive"] = False
    else:
        client.lease.controller.snapshot = lambda: {"owners": [client.test_owner, {"owner_id": "other"}]}
    with pytest.raises(RuntimeError, match="lane"):
        asyncio.run(client.warmup())
    assert client.model is None


def test_standalone_uses_real_extractor_validator_and_boot_signer_without_a_worker(client, monkeypatch):
    import mlx_lm.utils

    from core.brain import nonparametric_generation
    from core.brain.llm.latent_cortex import runtime_identity, worker_capture_identity
    from core.learning.semantic_program_feature_materialization import (
        validate_exclusive_lane_receipt,
    )
    from core.runtime import desktop_boot_safety

    calls = []
    model = SimpleNamespace(freeze=lambda: calls.append("freeze"), eval=lambda: calls.append("eval"))
    tokenizer = SimpleNamespace(encode=lambda text: [ord(character) for character in text])
    monkeypatch.setattr(desktop_boot_safety, "configure_mlx_process_device", lambda *_args, **_kwargs: {"verified": True})
    monkeypatch.setattr(mlx_lm.utils, "load", lambda *_args: (model, tokenizer))
    def identity(_model, **kwargs):
        capture = worker_capture_identity.validate_worker_capture_identity(kwargs["worker_action_capture_identity"])
        assert capture["worker_pid"] == os.getpid() and capture["worker_boot_id"] == client.boot_id
        assert kwargs["worker_source_path"].name == "mlx_worker.py"
        return {"worker_pid": os.getpid(), "worker_boot_id": client.boot_id,
                "worker_model_path": str(client.model_path), "worker_action_capture_identity": capture}
    monkeypatch.setattr(runtime_identity, "build_worker_identity", identity)
    class Encoder:
        def __init__(self, actual_model, actual_tokenizer):
            assert actual_model is model and actual_tokenizer is tokenizer
        def encode_hidden_sequence_ids(self, tokens):
            return np.asarray([[1., 0.] for _ in tokens], dtype=np.float32)
    monkeypatch.setattr(nonparametric_generation, "MLXEncoder", Encoder)
    async def acquire():
        assert await client.warmup()
        assert await client.warmup()
        before = client.get_model_lane_ownership_snapshot()
        validate_exclusive_lane_receipt(before, checkpoint=client.model_path)
        observation = await client.encode_hidden_sequence("ab", timeout_s=120.)
        assert observation["token_ids"] == [97, 98]
        assert observation["hidden_states"].shape == (2, 2)
        assert observation["receipt"]["model_basis"]["worker_pid"] == before["campaign_pid"]
        assert observation["receipt"]["generated_text"] is False
        client.test_owner["fencing_token"] += 1
        with pytest.raises(RuntimeError, match="ownership changed"):
            await client.encode_hidden_sequence("ab", timeout_s=120.)
        await client.aclose()
        assert client.get_model_lane_ownership_snapshot() == {}
    asyncio.run(acquire())
    assert calls == ["freeze", "eval"] and client.model is None and client.closed


def test_cleanup_drops_model_references_even_if_metal_barrier_fails(client, monkeypatch):
    import mlx.core as mx
    client.model = object()
    client.encoder_cache["encoder"] = object()
    monkeypatch.setattr(mx, "synchronize", lambda: (_ for _ in ()).throw(RuntimeError("barrier")))
    with pytest.raises(RuntimeError, match="barrier"):
        asyncio.run(client.aclose())
    assert client.model is None and not client.encoder_cache and client.closed


def test_response_validator_rejects_extractor_payload_tampering(client, monkeypatch):
    from core.brain.llm import mlx_worker
    client.model, client.tokenizer = object(), object()
    client.identity = {"worker_pid": os.getpid()}
    monkeypatch.setattr(mlx_worker, "_encode_hidden_sequence_response", lambda **_kwargs: {
        "token_ids": [1], "hidden_shape": [1, 2], "hidden_dtype": "float32_le",
        "hidden_state_bytes": b"bad", "receipt": {"hidden_size": 2}})
    with pytest.raises(RuntimeError, match="payload length"):
        asyncio.run(client.encode_hidden_sequence("source"))
    assert client.examples == 0


def test_timed_out_forward_cannot_become_accepted_evidence(client, monkeypatch):
    from core.brain.llm import mlx_worker
    from tools import semantic_feature_standalone
    client.model, client.tokenizer = object(), object()
    client.identity = {"worker_pid": os.getpid()}
    ticks = iter((0., 121.))
    monkeypatch.setattr(semantic_feature_standalone, "time",
                        SimpleNamespace(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(mlx_worker, "_encode_hidden_sequence_response", lambda **_kwargs: {})
    with pytest.raises(TimeoutError):
        asyncio.run(client.encode_hidden_sequence("source", timeout_s=120.))
    assert client.examples == 0
