# G03 v7 micro recovery, 2026-09-29

The frozen residual micro campaign reached one complete arm. Its three
reference requests generated three exact procedures and three correct public
answers, with no forced completion. The independent file verifier reports
`artifacts_verified=true` and no implementation drift. This measures a small
source cohort. It does not establish transfer or baseline preservation.

The next arm scored 673 alternatives for its first request and produced no
row after about 38 minutes. The original 3600-second whole-stage bound could
not cover all three requests at that rate. The supported detached supervisor
stopped the campaign. Its terminal receipt says `status=stopped`,
`returncode=-15`, `timed_out=false`, and confirms an empty process group and
lineage. This is an incomplete comparison, not a failed model answer.

Evidence is under `~/.aura/rlc-evidence/semantic-native-joint-graph-v7-residual-micro-v1-20260929/`.
The fitted arm's report and verification are at `reference/fitted/`; the
supervised stop is at `supervisor/detached_receipt.json`.

The next campaign can freeze an explicit `--max-seconds` of at most 14400
seconds. Resume rejects a changed budget. The default remains 3600 seconds.
This changes the time available to measure the same requests; it does not
weaken their exactness or control gates.

## Decision scoring

An opt-in scorer shares one full-shape model forward only when alternatives
have the same length, continuation boundary, scored positions, and every
causal predecessor token. It retains each original target token and computes
its own likelihood. It does not change the model, cache, training objective,
or search policy.

A detached 27B probe independently recomputed seven decisions from the first
two fitted requests. Forty-nine alternatives needed 19 group forwards. Direct
and shared scores agreed exactly for all 49 alternatives; the largest direct
and historical score difference was zero. The receipt reports a clean exit,
no timeout, and empty process group and lineage. This is a numerical check on
those decisions, not a wall-time measurement or a blanket equivalence claim.

Probe files are under `~/.aura/rlc-evidence/semantic-native-causal-groups-v3-20260929/`.
Its frozen plan SHA-256 is
`0a464807283486f22a465a08450c29b08f79de29c75a7d320fe40206ce791566`.
The opt-in grouped evaluator records its execution mode and per-decision
forward counts; the independent verifier reconstructs those counts from
source-bound sequences. The original individual scorer remains the default.

A one-request, source-matched grouped generation completed under
`~/.aura/rlc-evidence/semantic-native-grouped-one-v1-20260929/`. Its plan SHA-256 is
`e5ecdccc4401ed63a205b740e39868bd2b84a4bc143411a7e4d5c47f3e8d8875`.
The detached receipt reports a clean exit, no timeout, and empty process
group and lineage. Generation took 650.87 measured seconds and produced one
exact procedure and correct answer, with no forced completion. The independent
verifier reports `artifacts_verified=true`, no implementation drift, and
proven output/domain equivalence.

The first request is the same source as the earlier individual-scoring fitted
arm. A structured comparison, excluding only plan, receipt, and new execution
metadata, found no difference in the two rows. Every decision score, graph
score, selected program, search count, and answer verdict matches. Its 373
decision alternatives used 164 grouped forwards. Different run conditions
mean this is not a measured wall-time speedup.

The matched baseline and erasure arms have not completed. A fresh grouped
micro campaign is running under
`~/.aura/rlc-evidence/semantic-native-joint-graph-v7-residual-micro-v2-grouped-20260929/`.
Its `pipeline.json` and 24-command `broker_policy.json` were frozen before
model-active work. The supervisor plan SHA-256 is
`3243fb83e70190455a63532407d714605e78f1a2e1e60e18e258415d7145e7b3`.
This launch grants no baseline-preserving promotion or G03 closeout.
