"""Research state isolation cannot split the physical model-memory ledger."""

import os
import subprocess
import sys

from core.runtime import model_lane_control as lane
from core.runtime.state_ownership import RuntimeProfile


def test_live_model_lane_ignores_research_state_root(monkeypatch, tmp_path):
    private = tmp_path / "research"
    shared = tmp_path / "host"
    monkeypatch.delenv("AURA_MODEL_LANE_STATE_PATH", raising=False)
    monkeypatch.setattr(lane, "state_root", lambda: private)
    monkeypatch.setattr(lane, "live_state_root", lambda: shared)
    monkeypatch.setattr(lane, "runtime_profile", lambda: RuntimeProfile.LIVE)
    assert lane.ModelLaneController().state_path == shared / "run/model_lane_control.json"

    monkeypatch.setenv("AURA_MODEL_LANE_STATE_PATH", str(private / "explicit.json"))
    assert lane.ModelLaneController().state_path == private / "explicit.json"


def test_hermetic_model_lane_keeps_private_state_root(monkeypatch, tmp_path):
    private = tmp_path / "research"
    shared = tmp_path / "host"
    monkeypatch.delenv("AURA_MODEL_LANE_STATE_PATH", raising=False)
    monkeypatch.setattr(lane, "state_root", lambda: private)
    monkeypatch.setattr(lane, "live_state_root", lambda: shared)
    for profile in (RuntimeProfile.TEST, RuntimeProfile.BENCH, RuntimeProfile.DEV):
        monkeypatch.setattr(lane, "runtime_profile", lambda profile=profile: profile)
        assert lane.ModelLaneController().state_path == private / "run/model_lane_control.json"


def test_live_research_subprocess_shares_lane_without_sharing_evidence_state(tmp_path):
    environment = os.environ.copy()
    for name in ("AURA_MODEL_LANE_STATE_PATH", "AURA_TESTING", "AURA_PROOF_RUN",
                 "AURA_BENCH_RUN", "AURA_BENCHMARK", "AURA_DEV_RUNTIME"):
        environment.pop(name, None)
    environment.update({"AURA_HOME": str(tmp_path / "home"),
                        "AURA_LIVE_STATE_ROOT": str(tmp_path / "home/.aura"),
                        "AURA_STATE_ROOT": str(tmp_path / "research")})
    result = subprocess.run(
        [sys.executable, "-c", "from core.runtime.model_lane_control import ModelLaneController; "
         "from core.runtime.state_ownership import runtime_profile, state_root; "
         "print(runtime_profile().value); print(state_root()); "
         "print(ModelLaneController().state_path)"],
        capture_output=True, text=True, env=environment, timeout=20, check=True,
    )
    assert result.stdout.splitlines() == [
        "live", str(tmp_path / "research"),
        str(tmp_path / "home/.aura/run/model_lane_control.json"),
    ]
