# G03 joint graph checkpoint selection, 2026-09-28

The v7 source-only objective trains complete-graph contrasts alongside native
grammar path risk. Its first preparation still used the v5/v6 checkpoint rule:
it measured whole-graph rankings but selected only on exact teacher paths.
That could choose an unfitted or path-winning checkpoint while ignoring the
outcome the new objective was built to improve.

The v7 selection contract now admits a checkpoint only when it retains every
baseline-correct calibration source under **both** exact teacher-path and
positive whole-graph ranking. Among eligible checkpoints it prefers more
positive graph rankings, then more exact teacher paths, then lower calibration
loss. V5/v6 selection and their historical receipts are unchanged. These are
source-calibration observations, not a guarantee for held or generated work.

The new plan was prepared without loading model weights at
`/Users/bryan/.aura/rlc-evidence/semantic-native-joint-graph-v7-prep-selection-20260928`.
Its plan SHA-256 is
`2ff0aa483bfaca8f85997c6e7c39440c1effbded440816ec0525bc76c2923c22`;
the source-supervision receipt SHA-256 is
`3f841ef623a8c5738573bf911c407d505b4a6adeb9020488bb3816115851972f`.
The frozen 303-fit/185-calibration cohort yielded 23,685 lossless prefix
sequences, including 1,952 complete-graph sequences. The largest projected
float32 source shard is 223,457,280 bytes, below the declared 512 MiB bound.
No model-active fit, checkpoint gain, target-blind decode, or held transfer
has been measured for this plan.

Focused selection, checkpoint-reader, source-control, objective, search,
training and verification tests: 157 passed, two skipped. Smoke: 164 passed,
one skipped. Lint, compile, governance-lint, layering, writing and doc-drift
passed at their declared baselines. G03 remains open pending model-active fit,
independent replay, generated non-regression, and sealed transfer controls.
