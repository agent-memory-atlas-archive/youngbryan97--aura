# G03 source-fit handoff, 2026-09-30

## Build

`tools/run_semantic_source_handoff.py` continues the source-order bounded fit
into frozen utterance folds and the construction-disjoint proposer bank. It
reuses the existing detached supervisor and exact-command subprocess broker.
The source fit must terminate successfully with verified process cleanup
before either child starts. The source plan fixes the model, report, checkpoint
directory and full bundle inventory. No acquisition is repeated.

`core/learning/semantic_binary_fit_verification.py` reads each checksummed
optimizer archive and compares its vector with the head the serialized model
actually carries. The recorded fresh convergence, accepted iteration count,
final vector hash, zero center, float32 coefficient conversion and float64
bias must agree. A converged status alone is insufficient. The source report
must bind that model and the same archive inventory, report completed fitting,
and expose no test examples to fit.

The broker freezes two commands before the source fit completes: source-only
fold preparation and fold-zero proposer fitting. The latter has its own
binary checkpoints and disjoint fit, calibration and held constructions.
Every measured held row must appear in the final bank receipt. Missing output
or a nonzero, timed-out or uncontained child ends the handoff.

## Checks

The affected handoff, fold, proposer and native-stage suites completed with
57 passing tests in 18.78 seconds. Real miniature source fits exercise all
11 binary heads. Falsifications include missing archives, changed bytes,
rechecksummed wrong coefficients, false convergence, wrong iteration totals,
nonzero centers, partial inventories and an incomplete source report.
Command tests reject changed tools, solver changes, repeated outputs and
bundle identities, and missing option values.

## Boundary

Archive verification does not recompute the logistic objective, measure
semantic accuracy, select native weights, qualify a candidate or activate
serving. Utterance folds test wording transfer; they are not fresh semantic
families. Source-bank preparation supplies inputs for the separate native
fit and sequential decode gates. G03 and broader G-ledger requirements remain
open until their measured acceptance checks pass.
