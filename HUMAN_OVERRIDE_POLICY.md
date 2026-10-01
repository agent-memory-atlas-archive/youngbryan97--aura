# Human Override Policy — Aura Cognitive Runtime

## Principle

A human operator always has complete control over Aura. You can override, disable, or roll back any action or feature at any time. The system will never resist, bypass, or delay a human override command.

## Override Mechanisms

### 1. Immediate Kill

| Method | Effect | Data Loss Risk |
|--------|--------|----------------|
| Ctrl+C / SIGINT | Graceful shutdown (state saved) | None |
| SIGTERM | Graceful shutdown with 12s budget | None |
| SIGKILL | Immediate process death | Minimal (database write-ahead log recovery) |
| GUI close button | Graceful shutdown | None |
| `AURA_MODE=safe` | Disable all autonomous behavior | None |

### 2. Capability Disable

You can turn off any Aura capability at runtime:

```bash
# Disable autonomy
AURA_AUTONOMY_LEVEL=0

# Disable background tasks
AURA_FOREGROUND_ONLY=1

# Turn off a governed subsystem by its flag name
AURA_FLAG_WILL_STRICT_ENFORCEMENT=0
```

Flag names are defined in `_DEFAULT_FLAGS` within `core/governance/feature_flags.py`. To override a flag from your environment, add `AURA_FLAG_` to the capitalized flag name. Environment variables take precedence over both default settings and `feature_flags.json` under the state root.

To block specific tools or file paths, use **standing directives** rather than environment variables. Directives are saved to `data/governance/standing_directives.json`. The authority gateway checks this file on disk before running every important action. Because it reads the file directly from disk, prompt injection or conversational context cannot bypass it:

```python
from core.governance.standing_directives import add_directive, KIND_TOOL, SCOPE_ANY

add_directive(kind=KIND_TOOL, value="shell", reason="operator override", scope=SCOPE_ANY)
```

The directive file only supports blocklists (deny rules). It intentionally does not support allowlists (permit rules). If directives could grant permissions, a malicious prompt injection could create a permanent backdoor through the system's safety gate. You can block access to specific filesystem locations the same way using `KIND_PATH`.

There is no cloud fallback to disable because model inference (running AI predictions) is strictly local. See `docs/runbooks/local-inference-boundary.md`.

### 3. Memory Override

```bash
# Export all memories
make memory-export

# Delete a specific memory — through the app's memory controls, which call the
# POST /api/memory/delete API (interface/routes/memory.py)

# Delete all memories
make memory-purge

# Reset identity to canonical state
make identity-reset

# Restore from backup
make restore BACKUP=<path>
```

### 4. Governance Override

```bash
# Audit all Will receipts
python tools/receipt_coverage_validator.py --artifacts artifacts/current

# List all ungoverned actions (should be 0)
make governance-lint
```

Every action Aura takes produces a "Will receipt" — an append-only, tamper-evident audit log entry protected with cryptographic hashes. Because past receipts form an unchangeable audit trail, you cannot rewrite history. However, to withdraw authority for future actions, you can reset Aura's identity (`make identity-reset`) or revoke permissions for a paired device through the app (`POST /api/devices/revoke-scope`).

## Override Hierarchy

```
Admin Override → Operator Override → User Override → Will Decision → Subsystem
```

Higher levels in this chain always take precedence over lower levels. When an override is issued, the system obeys immediately without arguing.

## Override Logging

Every override action is permanently recorded in the audit trail with:
- Timestamp
- Override type
- Actor (user, operator, or administrator)
- Previous state
- New state
- Reason (if provided)

Overrides cannot be hidden from or erased from the audit logs.

## Non-Negotiable Rules

1. **Aura must never resist a shutdown command.**
2. **Aura must never hide its actions from the operator.**
3. **Aura must never circumvent permission restrictions.**
4. **Aura must always report its current capability state honestly.**
5. **Aura must always allow you to export or delete your stored memories.**
6. **Aura must always allow override logging to be read.**
7. **Override mechanisms must work even when Aura is degraded or malfunctioning.**
