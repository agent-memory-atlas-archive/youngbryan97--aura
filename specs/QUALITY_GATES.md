# Quality Gates

## Pre-commit checks (run automatically)

Every commit must pass these before merging:

1. **Syntax**: All Python files must be valid code with no syntax errors (`SyntaxError`).
2. **Tests**: `pytest tests/ -q` — all tests must pass.
3. **No hardcoded paths**: No personal home-directory references (such as `<USER_HOME>`) in git-tracked files.
4. **No model artifacts**: No large AI model weight files (such as `.safetensors` or `.gguf`) and no files larger than 1MB stored in git.
5. **No log files**: No `.log` files tracked in version control.
6. **Imports resolve**: All core Python modules must import without throwing errors.

## Per-feature checks

Before any new feature is considered complete:

1. **Unit tests exist**: tests must cover standard expected use (the happy path) and at least one error condition.
2. **Integration test**: verify that the feature works with the running live kernel (if applicable).
3. **No new stubs**: no unfinished placeholder functions that only contain `pass` or return `"not implemented"`.
4. **Personality preserved**: run a 3-turn test conversation and confirm the responses do not slip into generic AI assistant phrasing.
5. **Context window**: ensure the system prompt stays under 20,000 characters with the new feature active.

## Response quality gates (automated in benchmarks/)

1. **Generic marker count = 0**: no canned assistant openings or filler (such as "How can I help?", "Certainly!", etc.).
2. **Hedging marker count = 0**: no non-committal answers (such as "it depends" or "both are great"). Take a clear perspective.
3. **First-person usage > 0 per response**: every response must use first-person language ("I", "me", "my") to maintain an active personality.
4. **No raw metrics in response text**: internal scores (such as `valence=`, `arousal=`, or `phi=`) must never appear in chat dialogue.
5. **Memory recall**: the system must accurately recall a topic mentioned 5 conversation turns earlier.

## Long-horizon checks (benchmarks/long_horizon_stress.py)

1. **Drift detection**: generic phrases late in the conversation must not exceed 1.5 times the frequency in early turns.
2. **Identity persistence**: first-person language usage must not decline by more than 50% in the second half of the conversation.
3. **Substrate coherence**: internal background state values (mood, energy, and coherence) must remain stable, never producing `NaN` (invalid number) or extreme outlier values.
4. **Memory**: the system must still recall a topic from turn 5 when asked about it at turn 25.

## Deployment readiness

1. **INSTALL.md matches README** (specifying the same Python version and identical launch command).
2. **No author-specific paths**: confirm that no personal user paths are hardcoded in any tracked file.
3. **No daemon plists**: no background service files (`.plist`) configured with `RunAtLoad` or `KeepAlive`.
4. **All dependencies in requirements.txt**: every required package must be listed in `requirements.txt`.
5. **Fresh venv install succeeds**: running `pip install -r requirements.txt` on a clean machine inside a fresh virtual environment (`venv`) must complete without errors.
