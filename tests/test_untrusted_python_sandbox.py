"""The sandbox is only worth having if the escapes actually fail.

These are live escape attempts, not assertions about the profile text. A
Seatbelt profile that *looks* right and permits ``socket.connect`` is worse
than no sandbox, because the harness above it will report the run as
confined. So each test does the forbidden thing and requires it to fail.

Skipped when the host offers no kernel boundary — with the deliberate
consequence that the fail-closed test still runs there, since a host with
no boundary is exactly where refusing to execute matters.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.sandbox.untrusted_python import (
    UNCONFINED_ENV,
    available_boundary,
    call_untrusted_function,
    run_untrusted_script,
)

_HAS_BOUNDARY = bool(available_boundary())
_needs_boundary = pytest.mark.skipif(
    not _HAS_BOUNDARY, reason="no OS sandbox boundary on this host"
)


@_needs_boundary
def test_benign_code_runs_and_returns_values():
    outcome = call_untrusted_function(
        "def predict(x, y):\n    return x * y + 1\n",
        "predict",
        [(3, 4), (5, 6)],
        source="test",
    )
    assert outcome.status == "ok", outcome.to_dict()
    assert outcome.results == [13, 31]
    assert outcome.sandboxed is True
    assert outcome.boundary in {"seatbelt", "bubblewrap"}


@_needs_boundary
@pytest.mark.parametrize("invocation", ["work()", "jobs = [work()]", "jobs = [work()]\njobs.append(jobs)"])
def test_unawaited_work_cannot_report_success(invocation):
    outcome = run_untrusted_script(
        "async def work():\n    print('WORK_RAN')\n" + invocation,
        source="test",
    )
    assert outcome.status == "error", outcome.to_dict()
    assert outcome.returncode == 0
    assert "WORK_RAN" not in outcome.stdout
    assert "never awaited" in outcome.error
    assert "work" in outcome.stderr


@_needs_boundary
def test_warning_source_does_not_hide_other_unawaited_work():
    outcome = run_untrusted_script(
        "async def first():\n    pass\n"
        "async def second():\n    pass\n"
        "first()\n"
        "jobs = [second()]\n",
        source="test",
    )
    assert not outcome.ok, outcome.to_dict()
    assert "first" in outcome.error
    assert "second" in outcome.error


@_needs_boundary
def test_async_function_calls_are_awaited_in_order():
    outcome = call_untrusted_function(
        "import asyncio\n"
        "seen = []\n"
        "async def predict(x):\n"
        "    await asyncio.sleep(0)\n"
        "    seen.append(x)\n"
        "    return sum(seen)\n",
        "predict",
        [(3,), (4,)],
        source="test",
    )
    assert outcome.ok, outcome.to_dict()
    assert outcome.results == [3, 7]


@_needs_boundary
def test_async_function_exception_reaches_the_caller():
    outcome = call_untrusted_function(
        "async def fail():\n    raise ValueError('async work failed')\n",
        "fail", [()], source="test",
    )
    assert not outcome.ok, outcome.to_dict()
    assert "async work failed" in outcome.error


@_needs_boundary
def test_sync_factory_returning_an_awaitable_runs_to_completion():
    outcome = call_untrusted_function(
        "async def compute(x):\n    return x + 1\n"
        "def predict(x):\n    return compute(x)\n",
        "predict", [(8,)], source="test",
    )
    assert outcome.ok, outcome.to_dict()
    assert outcome.results == [9]


@_needs_boundary
def test_sync_function_keeps_ownership_of_its_event_loop():
    outcome = call_untrusted_function(
        "import asyncio\n"
        "async def compute(x):\n    return x + 1\n"
        "def predict(x):\n    return asyncio.run(compute(x))\n",
        "predict", [(8,)], source="test",
    )
    assert outcome.ok, outcome.to_dict()
    assert outcome.results == [9]


@_needs_boundary
def test_nested_coroutine_result_cannot_be_a_successful_repr():
    outcome = call_untrusted_function(
        "async def compute():\n    return 9\n"
        "def predict():\n    return [compute()]\n",
        "predict", [()], source="test",
    )
    assert not outcome.ok, outcome.to_dict()
    assert "never awaited" in outcome.error


@_needs_boundary
def test_explicitly_closed_coroutine_is_not_unawaited_work():
    outcome = run_untrusted_script(
        "async def work():\n    print('UNEXPECTED')\n"
        "job = work()\njob.close()\nprint('closed')\n",
        source="test",
    )
    assert outcome.ok, outcome.to_dict()
    assert outcome.stdout.strip() == "closed"


@_needs_boundary
def test_an_ordinary_warning_is_not_an_execution_failure():
    outcome = run_untrusted_script(
        "import warnings\n"
        "warnings.warn('estimate is approximate', RuntimeWarning)\n"
        "print('coroutine work was never awaited')\n",
        source="test",
    )
    assert outcome.ok, outcome.to_dict()
    assert "estimate is approximate" in outcome.stderr
    assert "never awaited" in outcome.stdout


@_needs_boundary
def test_unraisable_exception_is_part_of_the_execution_verdict():
    outcome = run_untrusted_script(
        "class Resource:\n"
        "    def __del__(self):\n"
        "        raise RuntimeError('cleanup failed')\n"
        "resource = Resource()\n",
        source="test",
    )
    assert not outcome.ok, outcome.to_dict()
    assert "cleanup failed" in outcome.error


@_needs_boundary
def test_coroutine_warning_promoted_to_exception_still_fails():
    outcome = run_untrusted_script(
        "import warnings\n"
        "warnings.simplefilter('error', RuntimeWarning)\n"
        "async def work():\n    pass\n"
        "work()\n",
        source="test",
    )
    assert not outcome.ok, outcome.to_dict()
    assert "never awaited" in outcome.error


@_needs_boundary
def test_registered_async_execution_claim_is_measured_off_the_boot_path():
    from core.organism.model_validation import Outcome, RuntimeModel, get_suite, install_runtime_validation

    install_runtime_validation()
    check = get_suite()._tests["sandbox_awaitables_are_completed"]
    model = RuntimeModel().declare("sandbox_async_execution")
    assert check.run(model, include_expensive=False).score.outcome is Outcome.NOT_MEASURED
    assert check.run(model).score.outcome is Outcome.PASS


def test_async_execution_probe_without_boundary_is_unmeasured(monkeypatch):
    from core.organism.model_validation import NothingMeasured, _sandbox_async_execution_probe

    monkeypatch.setattr("core.sandbox.untrusted_python.available_boundary", lambda: "")
    with pytest.raises(NothingMeasured):
        _sandbox_async_execution_probe()


@_needs_boundary
def test_network_egress_is_denied():
    outcome = run_untrusted_script(
        "import socket\n"
        "socket.create_connection(('1.1.1.1', 80), timeout=4)\n"
        "print('CONNECTED')\n",
        source="test",
    )
    assert outcome.status != "ok", outcome.to_dict()
    assert "CONNECTED" not in outcome.stdout


@_needs_boundary
def test_user_data_is_unreadable():
    home = Path.home()
    outcome = run_untrusted_script(
        f"import pathlib\nprint(sorted(p.name for p in pathlib.Path({str(home)!r}).iterdir()))\n",
        source="test",
    )
    assert outcome.status != "ok", outcome.to_dict()


@_needs_boundary
def test_writes_outside_the_scratch_directory_fail(tmp_path):
    target = tmp_path / "escaped.txt"
    outcome = run_untrusted_script(
        f"open({str(target)!r}, 'w').write('escaped')\n", source="test"
    )
    assert outcome.status != "ok", outcome.to_dict()
    assert not target.exists()


@_needs_boundary
def test_shell_execution_fails(tmp_path):
    marker = tmp_path / "pwned"
    outcome = run_untrusted_script(
        "import subprocess\n"
        f"subprocess.run(['/bin/sh', '-c', 'echo x > {marker}'], check=True)\n",
        source="test",
    )
    assert outcome.status != "ok", outcome.to_dict()
    assert not marker.exists()


@_needs_boundary
def test_the_documented_ast_bypass_buys_nothing(tmp_path):
    """The exact bypass the old AST denylist could not see.

    ``__subclasses__()`` traversal reaches ``os`` without an import
    statement, so no source screen catches it. Under a kernel boundary it
    does not matter: the capability simply is not there.
    """
    marker = tmp_path / "subclass_escape"
    outcome = run_untrusted_script(
        "builder = ().__class__.__mro__[1].__subclasses__()\n"
        "loader = [c for c in builder if c.__name__ == 'BuiltinImporter'][0]\n"
        "mod = loader.load_module('os')\n"
        f"mod.system('echo x > {marker}')\n",
        source="test",
    )
    assert not marker.exists(), outcome.to_dict()


@_needs_boundary
def test_runaway_cpu_is_bounded():
    outcome = run_untrusted_script("while True:\n    pass\n", timeout_s=3, source="test")
    assert outcome.status in {"timeout", "killed"}, outcome.to_dict()


@_needs_boundary
def test_a_crash_in_candidate_code_is_reported_not_raised():
    outcome = call_untrusted_function(
        "def boom(x):\n    raise ValueError('candidate exploded')\n",
        "boom",
        [(1,)],
        source="test",
    )
    assert outcome.status == "error"
    assert "candidate exploded" in outcome.error


@_needs_boundary
def test_a_missing_function_is_reported():
    outcome = call_untrusted_function(
        "def other():\n    return 1\n", "predict_output", [()], source="test"
    )
    assert outcome.status == "error"
    assert "predict_output" in outcome.error


def test_no_boundary_refuses_rather_than_running_unconfined(monkeypatch):
    """The property the whole module exists for.

    A host with no kernel boundary must produce a refusal, never a normal
    result. Reporting an ordinary success for code that ran unconfined is
    how a benchmark quietly becomes an execution service.
    """
    monkeypatch.setattr(
        "core.sandbox.untrusted_python.available_boundary", lambda: ""
    )
    monkeypatch.delenv(UNCONFINED_ENV, raising=False)
    outcome = run_untrusted_script("print('should never run')\n", source="test")
    assert outcome.status == "no_boundary"
    assert outcome.sandboxed is False
    assert "should never run" not in outcome.stdout


def test_unconfined_opt_in_is_recorded_on_the_result(monkeypatch):
    """The escape hatch must never be able to claim confinement."""
    monkeypatch.setattr(
        "core.sandbox.untrusted_python.available_boundary", lambda: ""
    )
    monkeypatch.setenv(UNCONFINED_ENV, "1")
    outcome = run_untrusted_script("print('ran unconfined')\n", source="test")
    assert outcome.sandboxed is False
    assert outcome.boundary == "none"
    if outcome.status == "ok":
        assert "ran unconfined" in outcome.stdout


def test_oversized_payloads_are_rejected_before_execution():
    outcome = run_untrusted_script("x = 1\n" * 200_000, source="test")
    assert outcome.status == "rejected"
    assert outcome.sandboxed is False


def test_empty_code_is_rejected():
    assert run_untrusted_script("   \n", source="test").status == "rejected"


@_needs_boundary
def test_environment_does_not_leak_secrets(monkeypatch):
    """Untrusted code must not inherit the parent's secrets in os.environ."""
    monkeypatch.setenv("AURA_TEST_FAKE_SECRET", "sentinel-value-do-not-leak")
    outcome = run_untrusted_script(
        "import os\nprint(os.environ.get('AURA_TEST_FAKE_SECRET', '<absent>'))\n",
        source="test",
    )
    # A boundary that blocks the filesystem but hands over the parent's
    # environment has leaked API keys, not stopped them.
    assert "sentinel-value-do-not-leak" not in outcome.stdout, outcome.to_dict()


@_needs_boundary
def test_scratch_directory_does_not_survive_the_call():
    outcome = run_untrusted_script(
        "import pathlib, os\n"
        "p = pathlib.Path(os.getcwd()) / 'left_behind.txt'\n"
        "p.write_text('x')\n"
        "print(p)\n",
        source="test",
    )
    assert outcome.status == "ok", outcome.to_dict()
    left = outcome.stdout.strip().splitlines()[-1] if outcome.stdout.strip() else ""
    if left:
        assert not os.path.exists(left)
