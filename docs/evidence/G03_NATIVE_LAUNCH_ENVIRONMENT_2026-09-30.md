# Native Handoff Launch Environment

Date: 2026-09-30

## Completed Upstream Work

The recovered source fit completed with terminal receipt
`51f124ed5ed2ab6f294bdc659031f7c927626a1b5da469d4f9d10a6278a8c4ee`.
All 11 archives independently match the serialized model. The original
500 validation identities remain unchanged; no test examples entered fit.

The recovered source-bank supervisor completed with terminal receipt
`8904c436d8759e622f6b7f115a9f3a71e8c9530837434e9cb939c4a1ccb7c022`.
Its parent, source report and frozen-fold identities independently verify.
Fit, calibration and bounded held inventories contain 1,335, 449 and 21
distinct identities, with no pairwise overlaps.

Bank report `f273662b27105ee5356f394e9f0667652b91ad70b0a1b3f5b1953af2643032e4`
contains all 21 held-row receipts. Correct interpretations are reachable
on 17/21; top joint scoring selects 15/21, compared with ordinary 16/21.
This diagnostic bank is not a promoted candidate or a general-transfer result.

## Failure and Repair

Native preparation failed before plan creation or model-weight loading. Its
supervisor inherited no `MLX_ENABLE_TF32` setting. The native float32/trie
execution contract correctly requires `MLX_ENABLE_TF32=0` at process launch.
The first brokered native-plan command refused that missing environment.
The dependent native-fit supervisor then refused its failed prerequisite and
launched no training.

Failed terminal receipts remain intact:

- Preparation: `dd03d5714448d3bb8c93de16706766dfcf76fa5f5d555e04b9db5aaaa3d98035`.
- Fit handoff: `fdaa05bb9f843b62edc5cd4c38b344a2a98e05a0d76feb3e49d8a941483a11d4`.

Both have verified containment and empty process groups. This failure is an
environmental launch defect, not a failed learned candidate.

`tools/run_semantic_source_handoff.py` now checks the declared precision and
prefix strategy through the existing CPU-only execution contract before
freezing native preparation jobs or waiting for prerequisites. The fit
handoff uses the same check. Missing, repeated or unsupported arithmetic
options fail explicitly. Neither check changes the caller's environment,
loads weights or loosens numerical tolerances.

Replacement launches must explicitly set `MLX_ENABLE_TF32=0` before starting
the detached supervisor, which freezes and propagates that value to brokered
children. They reuse the completed source fit and source bank; do not refit
either upstream stage. Preserve the failed runs and use new preparation and
fit-handoff directories. The native output directory remains
`/Users/bryan/.aura/rlc-evidence/semantic-native-stop-source-v1-20260930`.

## Checks and Boundary

The source-handoff, native-fit-handoff and arithmetic suites passed 79 tests
in 8.63 seconds. Tests exercise missing and wrong environment values,
explicit arithmetic custody and refusal before prerequisite waiting in both
normal and policy-only preparation. Native fitting, generated answers and
the promotion gates remain unmeasured for this candidate. G03 stays open.

Smoke passed 164 tests with one skipped in 40.83 seconds. Lint, compile,
governance and layering passed. Writing matched its baselines and document
drift found no broken references. Both real policy-only handoff rehearsals
passed with the explicit launch environment; neither loaded model weights.
