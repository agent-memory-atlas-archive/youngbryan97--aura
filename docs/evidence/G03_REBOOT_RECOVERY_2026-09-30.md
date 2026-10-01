# G03 Source Chain Reboot Recovery

Date: 2026-09-30

## Interruption

The host boot time is 10:14:38 PDT on September 30. All four original
supervisors and their children are dead, with no terminal receipts. Their
status is completion-indeterminate, not successful or a semantic failure.
The old plans, journals and logs remain unchanged.

Before recovery, the frozen target and broker execution manifests were
recomputed and matched their original identities. All ten numerical archives
passed checksum, identity, geometry and finite-value verification: nine
converged heads and one accepted iterate at iteration 475. The log ends at
450 because the next archive was saved before its progress line appeared.
No downstream broker job had started. Source candidate/report and native
training artifacts did not exist.

## Replacement Chain

All paths below are under `/Users/bryan/.aura/rlc-evidence`.

| Stage | Replacement Directory | Supervisor PID | Child PID |
| --- | --- | --- | --- |
| Source fit | `semantic-stop-source-fit-v4-recovery1-20260930` | 3642 | 3646 |
| Source bank | `semantic-source-handoff-v4-recovery1-20260930` | 3716 | 3724 |
| Native preparation | `semantic-native-preparation-v1-recovery1-20260930` | 3755 | 3759 |
| Native fit and verification | `semantic-native-fit-handoff-v1-recovery1-20260930` | 3788 | 3791 |

The source command retains its original identity:
`5a3c75c3c5cb3feab8144c3d48d0cf7b36886571f048392f08b9047696ce0644`.
It writes to the original `semantic-stop-source-fit-v4-checkpoint-20260930`
candidate/report paths and reads the original binary archives. Each head
must freshly converge under the unchanged objective and tolerances; a
stored converged flag alone grants no fitting authority.

The dependent commands change only supervisor and handoff artifact paths.
They use their original clean frozen checkouts. The replacement preparation
still writes native artifacts to `semantic-native-stop-source-v1-20260930`.
The replacement fit handoff waits for preparation and runs the exact prepared
fit followed by independent verification. Do not launch a second native fit.

Replacement plan identities, in stage order:

- `0df93f7fa8da3472ab1342c3e4203591360933927f8f25dc7582cffd4900d3da`
- `d096a47a66a0d2949c8f7b0f1904c3e19e730476e0eabf573da184138da732cf`
- `9fc4325adb5ad8592a664be2451719617b6d081544eaec575fae37eb7b607dff`
- `d3aeaabbdb8d18caa52959b486af56017de63a154788c5968b47962412187bb5`

Recovery evidence is in
`/Users/bryan/.aura/rlc-evidence/semantic-reboot-recovery-v1-20260930/recovery.json`,
with `launches.json` and `custody.json` beside it. It preserves pre-resume archive hashes, host boot time,
old plan identities and the replacement launch records. Replacement supervisors
have independent process groups, isolated logs/state and bounded execution.
They do not automatically retry numerical or scientific failures.

## Proof Boundary

Recovery does not close G03 or authorize serving. No generated evaluation
has occurred for this candidate. After terminal receipts, cleanup, selected
positive-step checkpoint and independent fit verification pass, the existing
16-node three-arm screen precedes the full 256-node reference, relation and
retained stages. Broader development and fresh transfer remain subsequent
obligations.

The recovery changes documentation only. Smoke passed: 164 tests, one
skipped, 88.92 seconds. Writing remained within its baselines; document
drift found no broken references. No model implementation was edited.
