# G03 v7 fit-matched residual, 2026-09-29

The completed v7 joint-graph fit selected its unfitted step-0 checkpoint.
Full-strength learned checkpoints lost source-calibration successes. Lower
teacher loss was not accepted as a substitute for baseline preservation.

The subsequent residual sweep measures checkpoint 101 at the predeclared
scales `0`, `0.0625`, `0.125`, `0.25`, `0.5`, and `1` on the same 185
source-calibration requests. No held or generated answer selects a scale.
The original training selection remains unchanged.

## Fit-matched replay

The earlier v3 residual report used a checkout with changed annotation bytes
in two fit-bound modules. The current micro verifier correctly refused that
lineage, despite the unchanged function bodies. Its report remains historical;
it was not re-sealed or granted an exception.

An isolated checkout at `2767ec8e4` restores the four fit-bound files to their
exact `ff53ad539` bytes. It retains the new residual decoder and verifier but
does not roll back those files on main. The frozen training verification and
all training implementation hashes match in that checkout. The fit-matched
source sweep was measured afresh there.

Evidence directory, relative to `~/.aura/rlc-evidence/`:
`semantic-native-joint-graph-v7-residual-step101-v4-fitmatched-20260929`.

- Calibration plan: `8450f44e398e9c3357a1fa7718a89fd73626b157b058c88f98909570723c4c55`.
- Report: `8c943c106246214707d1e1c8aa2d699050a802ea4961c765fc7ef65306a95adb`.
- The first `supervisor` attempt was refused before any model load or row by
  `exclusive_lane_requires_zero_owners`. The competing subject-core run was
  allowed to finish its 24 anchors and exit. This refusal is not a score.
- `supervisor-retry` completed with return code zero, no timeout, verified
  containment, empty process group and lineage, and a dead child. Its receipt
  is `729610fa515d513d819b67a03d3b473cca564f7a84bb2083bca691d1828e28de`.
- Model-active elapsed time was 1,872.63 seconds; supervised wall time was
  1,879.57 seconds. These are measured, not estimated.

Independent file replay selects scale `0.5`, with zero implementation drift:

| Source-calibration measure | Scale 0 | Scale 0.5 |
| --- | ---: | ---: |
| Exact teacher paths | 51/185 | 112/185 |
| Positive whole-graph rankings | 160/185 | 183/185 |
| Lost baseline-exact paths | 0 | 0 |
| Lost baseline-positive graphs | 0 | 0 |

There are 61 new exact paths and 23 new positive graph rankings. At scale
`0.5`, 90 of 868 reference decisions, 23 of 434 operation decisions, and one
of 434 termination decisions remain wrong. The first failing decision is a
reference in 53 sources, an operation in 19, and termination in one.

The independent verifier reconstructs receipt integrity, populations,
selection, and baseline preservation from files. It does not independently
recompute the model scores. Endpoint checks during the model-active sweep
reproduced the prior scale-0 and scale-1 measurements within the declared
`0.015625` score tolerance and required unchanged path/graph verdicts.

## Generated Micro Probe

The exact-command policy was frozen before this sweep completed in
`semantic-native-joint-graph-v7-residual-micro-v1-20260929/pipeline.json`
and `broker_policy.json`. It contains seven arms and 24 brokered commands.
The residual micro run is detached under its own supervisor. It advances
through the three reference requests, nine relation controls, and six retained
requests only after complete independent verification and each stage's exact
acceptance. Passed outputs are retained, not regenerated on continuation.

At this record, all evaluator plans have been materialized and the first
fitted reference arm is scoring its first request. There is no generated
success verdict yet. No 500-request run, G03 or G04 closeout, fusion, serving,
broad-reasoning gain, or frontier claim follows from the source-only sweep.

The source-matched checkout passed 163 focused checks, smoke (164 passed,
one skipped), lint, compile, governance-lint, and layering before launch.
These mechanical checks are not evidence of semantic transfer.
