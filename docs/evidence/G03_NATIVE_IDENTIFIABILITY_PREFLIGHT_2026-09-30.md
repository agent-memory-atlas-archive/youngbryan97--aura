# Exact-path identifiability before model loading, 2026-09-30

The native fitter can now run its existing causal-input collision audit before
loading model weights. `--require-identifiable-supervision` binds that requirement
into the training plan and writes `identifiability-preflight.json` before it
rejects conflicting targets. The plan binds both audit implementations. The
check runs on the captured fit and source-calibration partitions, separately
and together. Held labels do not enter it.
The independent fit verifier requires the saved preflight and recomputes it
from the bound supervision whenever the plan declares this strict requirement.

Two different exact targets on identical ordered scoring inputs cannot both
be selected by a deterministic scorer. Future unscored tokens and target
metadata cannot distinguish those inputs. The audit reports the maximum
number of exact teacher decisions attainable under this constraint. Its
strict requirement is opt-in: a probabilistic task can retain conflicting
observations without treating them as an invalid training set.

The immutable v7 supervision at
`semantic-native-joint-graph-v7-partial-final-20260928` was audited without
loading the resident model. All 4,416 decision groups have distinct scoring
inputs: 2,680 fit groups and 1,736 calibration groups. There are zero
contradictory scoring-input groups. The diagnostic receipt is
`5949de4942fe29149b321390eacafe54806dc04c0eeb23d24f20844463c7b3e5`, saved at
`semantic-native-identifiability-audit-20260930/report.json` under the RLC
evidence directory. The original artifacts were not changed.

This excludes one information-level impossibility in that supervision. It
does not establish that a finite-rank adapter can learn the targets, that
search reaches them, or that their meaning transfers. The new stop-source
fit and disjoint bank are still running; their native supervision has not
been audited. G03 remains open.

The affected audit, preparation, grammar-supervision, fit-sampling,
prefix-reuse, checkpoint-reader and source-control suites passed 119 tests
with two skips. They include a conflict visible only across the two source
partitions, durable evidence before rejection, stale receipts, held-source
contamination, altered preflight proofs, and compatibility between in-memory
and serialized partitions.
