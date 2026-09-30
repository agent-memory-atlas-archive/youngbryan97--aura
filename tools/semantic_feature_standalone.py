"""Run the resident feature extractor in one exclusively owned offline process."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path


def wait_for_lane_release(run_directory, *, plan_sha256, timeout, inspect=None,
                          clock=time.monotonic, sleep=time.sleep):
    """Require contained process cleanup, irrespective of the experiment's score."""
    if (type(timeout) not in {int, float} or not math.isfinite(timeout)
            or not 0 < timeout <= 86400 or not isinstance(plan_sha256, str)
            or len(plan_sha256) != 64
            or any(character not in "0123456789abcdef" for character in plan_sha256)):
        raise ValueError("model-lane wait requires a frozen plan and finite bound")
    if inspect is None:
        from tools.run_detached_step import _status
        inspect = _status
        run_directory = Path(run_directory)
    deadline = clock() + timeout
    while True:
        status = inspect(run_directory)
        if status.get("plan_sha256") != plan_sha256:
            raise ValueError("model-lane predecessor plan changed")
        if status.get("terminal") is True:
            receipt = status.get("receipt") or {}
            if (receipt.get("plan_sha256") != plan_sha256
                    or receipt.get("containment_verified") is not True
                    or status.get("child_state") != "dead"):
                raise ValueError("model-lane predecessor lacks proven process cleanup")
            return receipt
        if status.get("supervisor_alive") is not True or status.get("completion_indeterminate") is True:
            raise ValueError("model-lane predecessor cannot prove continued execution")
        remaining = deadline - clock()
        if remaining <= 0:
            raise ValueError("model-lane wait reached its bound; no model was loaded")
        print(json.dumps({"stage": "model_lane_wait", "state": status.get("state"),
                          "plan_sha256": plan_sha256,
                          "heartbeat_sequence": status.get("heartbeat_sequence")}), flush=True)
        sleep(min(30., remaining))


class StandaloneSemanticFeatureClient:
    """Use the unchanged extractor and response validator without spawning a worker."""

    def __init__(self, model_path, *, lease, envelope):
        self.model_path = Path(model_path).resolve(strict=True)
        self.lease = lease
        self.envelope = envelope
        self.model = None
        self.tokenizer = None
        self.identity = {}
        self.encoder_cache = {}
        self.closed = False
        self.examples = 0
        self.boot_id = uuid.uuid4().hex
        from core.runtime.lockdep import checked_lock
        self.metal_lock = checked_lock("semantic_feature_standalone.metal")

    def get_worker_identity_snapshot(self):
        return copy.deepcopy(self.identity)

    def _owner_decision(self):
        lease = self.lease
        decision = lease.decision
        if (not lease.active or lease.inherited or lease.controller is None or decision is None
                or not decision.admitted or not decision.ready_to_spawn
                or decision.state.value != "committed" or not decision.receipt_id
                or decision.fencing_token <= 0):
            raise RuntimeError("standalone features require a committed independent lane fence")
        if not lease.controller.heartbeat_owner_sync(
                decision.owner_id, fencing_token=decision.fencing_token):
            raise RuntimeError("standalone feature lane fence was lost")
        owners = lease.controller.snapshot()["owners"]
        matching = [row for row in owners if row.get("owner_id") == decision.owner_id]
        if (len(owners) != 1 or len(matching) != 1
                or matching[0].get("exclusive") is not True
                or matching[0].get("fencing_token") != decision.fencing_token
                or matching[0].get("process", {}).get("pid") != os.getpid()
                or matching[0].get("model_path") != str(self.model_path)):
            raise RuntimeError("standalone feature lane ownership changed")
        return decision

    def get_model_lane_ownership_snapshot(self):
        if self.closed or not self.identity:
            return {}
        decision = self._owner_decision()
        body = {"schema": "aura.mlx_model_lane_ownership.v1", "exclusive": True,
                "owner_id": decision.owner_id, "fencing_token": decision.fencing_token,
                "terminal_receipt_id": decision.receipt_id,
                "model_path": str(self.model_path), "campaign_pid": os.getpid(),
                "worker_pid": os.getpid(), "worker_boot_id": self.boot_id}
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("ascii")
        return {**body, "receipt_sha256": hashlib.sha256(encoded).hexdigest()}

    async def warmup(self, **_kwargs):
        if self.closed:
            raise RuntimeError("standalone feature client is closed")
        if self.model is not None:
            self._owner_decision()
            return True
        self._owner_decision()
        from core.runtime.desktop_boot_safety import configure_mlx_process_device
        device = configure_mlx_process_device("metal", reason="standalone_semantic_features", force=True)
        if device.get("verified") is not True:
            raise RuntimeError("standalone feature Metal device is unverified")
        from mlx_lm.utils import load

        from core.brain.llm import mlx_worker
        from core.brain.llm.latent_cortex.runtime_identity import build_worker_identity
        from core.brain.llm.latent_cortex.worker_capture_identity import (
            build_worker_capture_identity,
        )
        print(json.dumps({"stage": "feature_model_load", "model_path": str(self.model_path),
                          "execution_mode": "standalone_no_fork_v1", "pid": os.getpid()}), flush=True)
        self.model, self.tokenizer = load(str(self.model_path))
        self.model.freeze()
        self.model.eval()
        capture = build_worker_capture_identity(worker_boot_id=self.boot_id)
        self.identity = build_worker_identity(self.model, model_path=self.model_path,
            worker_boot_id=self.boot_id, worker_source_path=Path(mlx_worker.__file__),
            worker_action_capture_identity=capture.public_identity, tokenizer=self.tokenizer)
        self._owner_decision()
        return True

    async def encode_hidden_sequence(self, text, *, timeout_s=8., representation="final_hidden_v1"):
        if (type(timeout_s) not in {int, float} or not math.isfinite(timeout_s) or timeout_s <= 0):
            raise ValueError("standalone feature forward requires a finite timeout")
        if self.closed or self.model is None or not self.identity:
            raise RuntimeError("standalone feature client is not ready")
        self._owner_decision()
        from core.brain.llm.mlx_latent_reasoning import _ReasonsInLatentSpace
        from core.brain.llm.mlx_worker import _encode_hidden_sequence_response
        started = time.monotonic()
        response = _encode_hidden_sequence_response(model=self.model, tokenizer=self.tokenizer,
            text=text, request_id=uuid.uuid4().hex, encoder_cache=self.encoder_cache,
            worker_identity=self.identity, metal_semaphore=self.metal_lock,
            representation=representation)
        elapsed = time.monotonic() - started
        # A synchronous Metal call cannot be cancelled safely. The OS supervisor
        # bounds a hung process; an overrun that returns is excluded from evidence.
        if elapsed > timeout_s:
            raise TimeoutError("standalone feature forward exceeded its declared bound")
        self._owner_decision()
        observation = _ReasonsInLatentSpace._validate_hidden_sequence_response(
            self, text, response, representation=representation)
        self.examples += 1
        self.envelope.reclaim(self.examples)
        print(json.dumps({"stage": "feature_forward", "completed_forwards": self.examples,
                          "elapsed_seconds": elapsed,
                          "source_sha256": hashlib.sha256(text.encode()).hexdigest()}), flush=True)
        return observation

    async def aclose(self):
        if self.closed:
            return
        self.closed = True
        import mlx.core as mx
        try:
            mx.synchronize()
        finally:
            self.encoder_cache.clear()
            self.model = None
            self.tokenizer = None
            self.identity.clear()
            gc.collect()
        mx.clear_cache()
        mx.synchronize()


def acquire_standalone_jobs(*, model, tokenizer, tokenizer_identity, jobs):
    """Acquire the lane outside asyncio, then run one model lifecycle for all cohorts."""
    import asyncio

    from core.runtime.desktop_boot_safety import configure_mlx_process_device
    from core.runtime.mlx_memory_guard import mlx_memory_envelope
    from core.runtime.model_lane_control import standalone_model_lane
    from tools.materialize_semantic_program_features import _acquire_jobs

    with (standalone_model_lane(owner_id="semantic-feature-materialization",
            model_path=str(model), purpose="evaluation", require_exclusive=True,
            allow_owner_eviction=False, preemptible=False,
            metadata={"production_effect": False, "feature_execution_mode": "standalone_no_fork_v1"}) as lease):
        device = configure_mlx_process_device("metal", reason="standalone_semantic_features", force=True)
        if device.get("verified") is not True:
            raise RuntimeError("standalone feature Metal device is unverified")
        with mlx_memory_envelope(fraction=.80) as envelope:
            client = StandaloneSemanticFeatureClient(model, lease=lease, envelope=envelope)
            return asyncio.run(_acquire_jobs(client, model=model, tokenizer=tokenizer,
                tokenizer_identity=tokenizer_identity, jobs=jobs))
