# G03 binary iterate resume, 2026-09-30

## Cause and Repair

The completed feature acquisition remains valid. The subsequent source fit
failed because its fifth binary pointer head exhausted a caller's 250-iteration
limit. Four earlier heads had converged, but the run saved no coefficients.
The [terminal record](G03_STOP_SOURCE_REACQUISITION_2026-09-30.md#source-fit-terminal-result)
preserves that failure; it is not a candidate.

The opt-in bounded solver now has a 1,000-iteration caller bound for each binary
pointer, argument-role, argument-proposal and directional-relation head. The
weighted objective, feature construction, bias regularization and original
gradient/function tolerances are unchanged. Ordinary liblinear bounds are
unchanged. Exhausting the new bound still fails; no unconverged head is returned.

## Existing Checkpoint Owner

The repair reuses `ObjectiveFitCheckpoint` in
`core/learning/semantic_fit_checkpoint.py`, including its checksummed numerical
archive and file-write gateway. A checkpoint scope binds a caller's source or
partition custody. Each objective key hashes every feature row, labels, sample
weights, shape, tolerances, objective contract, implementation and dependency
versions. Feature hashing uses the same bounded 256-row materialization as
optimization; it does not expand a full matrix.

Every 25 accepted optimizer iterations are retained. Finite terminal iterates
are saved even when an iteration bound prevents convergence. A saved iterate
initializes a fresh L-BFGS history against the unchanged complete objective.
Even an archive marked converged must pass a new solver convergence check.
Changed rows, labels, weights, tolerances or source custody cannot reuse it.

Both source-fitting and construction-disjoint proposer CLIs accept
`--binary-checkpoints` with `blocked_lbfgs`. Native or held targets do not enter
the cache. Source fitting records the actual checkpoint/convergence inventory
in the model and report. This is numerical work preservation, not selection,
qualification, model publication or serving authority.

## Checks and Boundary

`tests/test_semantic_binary_fit_resume.py` exercises iteration exhaustion,
real optimizer interruption, warm-start equivalence, checkpoint corruption,
changed objective/partition keys, bounded materialization, nested scopes and
receipt immutability. A converged archive is explicitly denied authority when
the next solver call does not converge.

`tests/test_compositional_source_training.py` fits and round-trips all 11 binary
heads with actual archives, unchanged split custody and no test examples.
These checks do not prove that the larger source fit will converge or that
its native generation will pass development. G03 remains open. The generic
detached supervisor's resume contract is separate from these numerical archives;
this repair does not claim a supervisor-level restart verifier.

The seven affected fitter/custody suites completed with 108 passing tests in
53.98 seconds. A separate fold-preparation/resume pass completed 23 tests in
5.81 seconds. Those counts overlap and are not added together as new tests.
