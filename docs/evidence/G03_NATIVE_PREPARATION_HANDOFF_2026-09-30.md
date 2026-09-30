# Source bank to native preparation, 2026-09-30

The source handoff can now wait for the existing source-bank supervisor and
prepare the native fit. It does not repeat acquisition, source fitting,
fold preparation, or bank fitting. The bank's original supervised command
must refer to the same source-fit supervisor. Its terminal receipt must prove
successful completion and process cleanup. The bank must bind the source
candidate, source report, folds, and every declared held-row receipt.

Two brokered commands then run the native trainer in plan-only and
supervision-only modes. They use the same protocol: 303 updates, checkpoints
every 101 updates, a rank-eight final-layer adapter, float32 trie execution,
lossless source shards, a 512 MiB resident shard bound, role-relative
registers, complete typed source pairs, path risk, and four whole-graph
contrasts. Construction/depth-balanced sampling keeps the partial epoch
explicit. The new exact-path identifiability requirement runs before any
model weights can load.

`native-preparation.json` binds the two upstream terminal receipts, the
verified source fit, native plan, supervision, and independently rechecked
identifiability preflight. Neither a preparation receipt nor a zero process
exit qualifies a candidate. Model fitting, checkpoint verification,
target-blind decode, controls, development, fresh transfer, public answers,
live activation, fusion, and broader claims remain separate stages.

The affected handoff, preflight, native-stage, fold and proposer suites passed
81 tests in 27.99 seconds. They check the common preparation protocol,
source-supervisor custody, stale candidate/report/fold bytes, missing held
rows, and a bank belonging to another parent. Full-cohort native preparation
has not completed. G03 remains open.
