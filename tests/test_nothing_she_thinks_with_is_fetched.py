"""She could not reply with the network off, and a file was the reason.

`get_model_path` is the serving resolver: every lane, every worker respawn, the
live learner and the optimizer ask it where the weights are. When an artifact was
missing locally it answered a Hugging Face repository id — for any model through
`HF_FALLBACKS`, and for the cortex itself through `_LEGACY_CORTEX_REPOSITORY_ID`.
That id goes to `mlx_lm.load`, which calls `snapshot_download` for anything that
is not a local directory. So with the wifi off the resolver blocked on DNS, the
lane never reached `ready`, `check_health()` was false for every lane, and
routing had nowhere to send the turn: no reply at all, because a file was not
where she looked.

Nothing she needs in order to think may depend on the internet.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.brain.llm import model_registry as registry


# ── the serving resolver ─────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(registry.HF_FALLBACKS))
def test_no_model_resolves_to_something_that_has_to_be_downloaded(name):
    resolved = registry.get_model_path(name)
    assert not registry.is_model_repository_id(resolved), resolved
    assert "/" in resolved or resolved == name, "an answer a caller can print"


def test_a_missing_model_answers_the_path_it_looked_in(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "get_models_dir", lambda: tmp_path)
    monkeypatch.setattr(
        registry, "_configured_model_location", lambda name: tmp_path / name
    )
    resolved = registry.get_model_path("Qwen2.5-72B-Instruct-4bit")
    assert resolved == str(tmp_path / "Qwen2.5-72B-Instruct-4bit")
    assert not registry.is_model_repository_id(resolved)


def test_a_missing_model_is_recorded_rather_than_raised(tmp_path, monkeypatch):
    """A caller asking where a model is has a right to an answer it can print."""
    from core.runtime.errors import get_degradation_tracker

    tracker = get_degradation_tracker()
    tracker.reset()
    monkeypatch.setattr(registry, "get_models_dir", lambda: tmp_path)
    monkeypatch.setattr(
        registry, "_configured_model_location", lambda name: tmp_path / name
    )
    registry.get_model_path("Qwen2.5-32B-Instruct-4bit")
    recent = tracker.recent(subsystem="model_registry", limit=1)
    assert recent
    assert "repository id" in str(recent[0].action)


def test_the_cortex_answers_a_path_even_when_it_is_not_there(tmp_path, monkeypatch):
    """The cortex used to resolve to a remote repository of its own."""
    absent = tmp_path / "no-cortex-here"
    monkeypatch.setattr(registry, "_current_cortex_path", lambda: absent)
    resolved = registry.get_model_path(registry.CORTEX_LOGICAL_NAME)
    assert resolved == str(absent)
    assert not registry.is_model_repository_id(resolved)


def test_a_local_artifact_still_resolves_to_itself(tmp_path, monkeypatch):
    present = tmp_path / "Qwen3.5-9B-4bit"
    present.mkdir()
    monkeypatch.setattr(registry, "get_models_dir", lambda: tmp_path)
    monkeypatch.setattr(
        registry, "_configured_model_location", lambda name: tmp_path / name
    )
    assert registry.get_model_path("Qwen3.5-9B-4bit") == str(present.resolve())


def test_the_fallback_table_is_kept_for_the_explicit_fetch():
    """`HF_FALLBACKS` still names what a download would fetch; it just is not
    the answer the serving resolver gives."""
    assert registry.HF_FALLBACKS
    assert all(registry.is_model_repository_id(v) for v in registry.HF_FALLBACKS.values())


# ── the load seam ────────────────────────────────────────────────────────


def test_the_worker_refuses_to_serve_from_anything_but_a_directory():
    """The seam that makes sure nothing else can reintroduce a fetch."""
    import inspect

    from core.brain.llm import mlx_worker

    source = inspect.getsource(mlx_worker)
    marker = "a serving load needs a local model directory"
    assert marker in source
    guard = source[source.index(marker) - 400 : source.index(marker)]
    assert "os.path.isdir" in guard


def test_the_only_downloader_has_no_caller_in_the_runtime():
    """Fetching is an explicit operation, not something a turn can trigger."""
    import subprocess

    found = subprocess.run(
        ["grep", "-rn", "ensure_present(", "--include=*.py", "core", "interface"],
        capture_output=True,
        text=True,
    )
    callers = [
        line
        for line in found.stdout.splitlines()
        if "def ensure_present(" not in line
    ]
    assert callers == [], callers


# ── and the name that cost the diagnosis ─────────────────────────────────


def test_nothing_reaches_ollama():
    """There has been no Ollama in the path for months.

    The word survived in four docstrings, and the file that named an "Ollama
    bridge" is the first place anyone looks when the network goes down — it cost
    a diagnosis. What the tree must not contain is a way to reach it: its port,
    its client, or a URL pointing at it.
    """
    import subprocess

    found = subprocess.run(
        [
            "grep",
            "-rn",
            "-E",
            r"11434|import ollama|from ollama|ollama\.(chat|generate|Client)|//[^\"\']*ollama",
            "--include=*.py",
            "core",
            "llm",
            "interface",
        ],
        capture_output=True,
        text=True,
    )
    assert found.stdout.strip() == "", found.stdout


def test_the_legacy_shim_does_not_claim_a_bridge_it_does_not_have():
    from llm import client

    assert "Ollama bridge" not in (client.OpenAIClient.__doc__ or "")
    assert "resident MLX brain" in (client.OpenAIClient.__doc__ or "")
