# Aura Tool Use Policy

*Reviewed against the tree: 2026-08-01. See [documentation status map](docs/DOC_STATUS.md) for how to read this file.*

## Scope

This policy applies to every tool and skill executed within Aura.

The core rule is simple: running a tool is an action with real-world consequences. It is never treated as just an ordinary function call. Because tools interact with files, systems, and networks, every tool run must be authorized, isolated in a sandbox, logged in an audit trail, and recoverable before it runs — not justified after the fact.

This rule applies universally. There is no special "safe" category of tools that skips security checks. Even skills that Aura writes automatically during self-repair must go through the exact same checks as skills requested by a human user.

## Principles

1. **No tool runs without Will authorization:** Every tool call must pass through Aura's decision engine (the Unified Will) and generate a signed audit record (`WillReceipt`).
2. **All tool output is untrusted:** Data returned by any tool is treated as external, untrusted input. It is sanitized and checked before it can influence Aura's decisions.
3. **Least privilege:** Skills only receive the minimum permissions needed to do their specific job.
4. **Fail closed (default to deny):** If the authorization system is unavailable or cannot make a decision, the tool is blocked from running.
5. **Full auditability:** Aura logs every tool call, including its authorization, inputs, outputs, and final results.

## Skill Contract

Every skill and tool must declare a manifest defining its permissions and requirements:

```yaml
name: skill_name
version: "1.0.0"
description: "What this skill does"
risk_level: low | medium | high | critical
permissions:
  filesystem: none | read | write | workspace_only
  network: none | local | external
  shell: none | sandboxed | full
  memory: none | read | write
input_schema:
  type: object
  properties: { ... }
output_schema:
  type: object
  properties: { ... }
timeout_s: 30
max_memory_mb: 512
sandbox_policy: strict | permissive | none
audit_policy: full | summary | none
owner: "author name"
tests: "tests/test_skill_name.py"
```

## Permission Matrix

### By Role

| Permission | User | Operator | Admin | Research |
|-----------|------|----------|-------|----------|
| Chat | ✅ | ✅ | ✅ | ✅ |
| Read tools (clock, weather) | ✅ | ✅ | ✅ | ✅ |
| File tools (workspace only) | ✅ | ✅ | ✅ | ✅ |
| File tools (outside workspace) | ❌ | ✅ | ✅ | Sandbox |
| Shell (sandboxed) | ❌ | ✅ | ✅ | Sandbox |
| Shell (unrestricted) | ❌ | ❌ | ✅ | ❌ |
| Browser | Limited | ✅ | ✅ | Sandbox |
| Network (external) | ❌ | ✅ | ✅ | Sandbox |
| Memory read | Own | All | All | Sandbox |
| Memory write | Own | All | All | Sandbox |
| Memory delete | ❌ | ❌ | ✅ | ❌ |
| Self-repair | ❌ | Approve | ✅ | Sandbox |
| Plugin install | ❌ | ❌ | ✅ | ❌ |
| Model change | ❌ | ✅ | ✅ | ✅ |
| Feature flags | ❌ | Limited | ✅ | Limited |
| Cloud fallback | ❌ | ✅ | ✅ | ❌ |

### By Risk Level

| Risk Level | Authorization | Sandbox | Audit | Example |
|-----------|---------------|---------|-------|---------|
| Low | Auto-approve | Optional | Summary | Clock, calculator |
| Medium | Will decision | Recommended | Full | File read, web search |
| High | Will + operator confirm | Required | Full | Shell exec, file write |
| Critical | Will + admin confirm | Required + isolated | Full | Self-modification, plugin install |

## Operator Controls

Tool permissions are managed by writing block rules (prohibitions) rather than allowlists. These rules, called standing directives, are stored in `data/governance/standing_directives.json`. Aura's authority gateway reads this file directly from disk before executing any action:

```python
from core.governance.standing_directives import (
    add_directive, remove_directive, KIND_TOOL, KIND_PATH, SCOPE_ANY, SCOPE_WRITE,
)

add_directive(kind=KIND_TOOL, value="shell", reason="operator policy", scope=SCOPE_ANY)
add_directive(kind=KIND_PATH, value="~/Documents", reason="off limits", scope=SCOPE_WRITE)
```

- `SCOPE_WRITE` blocks write or modify operations while still allowing read access.
- `SCOPE_ANY` blocks both read and write access completely.
- `remove_directive(directive_id)` deletes a directive to restore access.

The directive system only supports blocking actions, not granting them. This deliberate design prevents security vulnerabilities: if directives could grant permissions, an attacker using prompt injection could manipulate the AI into giving itself permanent privileges. Because the system only recognizes bans, any tampered file can only restrict functionality, never expand it.

If the directives file exists but cannot be read (for example, due to a disk error or corruption), Aura blocks all actions except read-only operations and logs a degraded status. It knows restrictions exist, so it errs on the side of caution.

Broader system settings are controlled through feature flags: setting `AURA_FLAG_<NAME>` overrides any flag defined in `_DEFAULT_FLAGS` (`core/governance/feature_flags.py`). For example, `workspace_jail_enabled` prevents file tools from escaping the designated workspace folder, and `will_strict_enforcement` controls whether the Will's security decisions are mandatory or advisory.

## Production Mode Rules

When running in production mode (`AURA_MODE=production`):
- Skills without a valid manifest or signature will not load.
- Self-modification tools are completely disabled.
- Running shell commands requires operator approval.
- Network access requires explicit manual configuration.
- All tool outputs are sanitized before Aura processes them.
- Tool timeouts are strictly enforced to prevent hanging processes.
