# G03 bounded native fit, 2026-09-30

The early-stop corpus adds 1,944 training-only requests to the eight existing
source cohorts. An ordinary short epoch would let this larger cohort dominate
the pilot, while requiring every source as a primary update would change its
runtime substantially. The new opt-in `construction_depth_balanced_v1` schedule
uses Aura's existing recurrent-SFT sampler to balance construction/depth strata.
The default native schedule is unchanged.

Each scheduled source keeps all its witnessed typed decision contrasts. Its
donors can be unscheduled sources, but must belong to the admitted fit partition.
The plan records primary exposure separately from frozen-prefix capture. The
independent verifier reconstructs the schedule and pairs, and binds fit,
calibration, held, captured and scheduled identities to the source bank.
Matching the bank's hashes alone no longer suffices to change those partitions.
The previous complete 303-source native plan passes this added partition check.

Calibration for this pilot samples by construction **and depth**, using source
identity rather than outcomes. This keeps early-stop and late-stop programs from
being lost when they share a construction. Optional global source limits remain
explicit canary limits; they do not establish complete calibration coverage.

## Metadata Rehearsal

The receipt is
`~/.aura/rlc-evidence/semantic-native-stop-fit-preflight-20260930/receipt.json`,
with receipt SHA-256 `e670eb67e2c67c31842dc49dfb1edfe2147f667d2aae82b1f6d7619d4c87a429`.
The rehearsal rebuilt source metadata, used the active 27B tokenizer, rebound
input registers to source order, and froze utterance fold 0. It read no hidden
arrays and loaded no model weights.

- Fit: 1,335 sources; calibration: 449; held: 924.
- Updates: 303 across 36 construction/depth strata, each visited 8 or 9 times.
- Distinct primary sources: 189; captured sources including fit-only donors: 608.
- Paired primary sources: 189; paired updates: 303.
- Sources with operation/reference/termination contrasts: 189/123/85.
- Interaction updates by those kinds: 638/279/141.
- First possible stop, decision 3: 85 paired sources. Late stop, decision 11: 56.
- Calibration: 11 source-identity samples, retaining stop-corpus depths 1, 3 and 4.

Schedule receipt SHA-256:
`2877e9c8c8d7a5611c07ae4ec0f0392aa48b04d14eed3ec083d0bd41f2b27d56`.
The schedule is a partial epoch. Capturing 608 sources does not mean 608 primary
updates happened. The held partition tests wording transfer, not fresh semantic
families. New neural feature acquisition and training remain unmeasured here.

## Verification

Focused checks: 110 passed, two skipped. Smoke: 164 passed, one skipped.
Lint, compile, governance lint, layering and writing passed. Tests cover changed
exposure, schedule order, source metadata, calibration sampling, foreign donors,
and attempts to move bank-held sources into training. The typed source-control
fixture now includes its required decision index.

The previous native micro campaign still owns the model lane and is completing
its erasure arm. This record grants no learned-gain, promotion, fusion or serving
authority. G03 remains unchecked.
