# G03 bounded source-head fitting, 2026-09-30

## Execution Repair

The expanded feature acquisition has 3,708 requests. A first-record projection
across its nine cohorts gives 19.738 GiB of hidden-state arrays; this is a sample
projection, not a measured full-population footprint. Dense source-head fitting
also stacks repeated pointer rows and expands every directional relation into
product, absolute-difference and signed-difference features.

The opt-in `blocked_lbfgs` execution path keeps ordered pointer-row references
and reuses `DirectionalFeatureRows` for relation features. Each objective
evaluation visits every row in batches of at most 256. There is no sampling,
feature projection, quantization or discarded alternative. The default
`liblinear` path is unchanged.

The objective is the existing balanced, sample-weighted logistic loss at C=10,
plus the L2 penalty on the weights and unit bias feature. The installed
scikit-learn class-weight function supplies the exact balancing convention.
Its current implementation uses sample weights when computing class balance.
The first weighted comparison exposed the difference from unweighted class
counts; the implementation was corrected rather than relaxing the comparison.

SciPy's L-BFGS-B solver must report convergence with finite parameters and
objective. An unfinished fit is refused. Source and crossfit entry points can
request this execution path. Its contract enters the fitted model, source
report and crossfit plan identities. A reused crossfit model must match the
declared execution. Progress records name starts, actual iteration counts and
completed fits.

## Proof Boundaries

The analytic objective and gradient match a full-matrix calculation and
independent finite differences. Weighted and unweighted fitted coefficients
match the existing solver within the declared tolerances, including its
regularized bias. A 600-row directional-feature test rejects any request for
more than 256 rows and still matches the dense fit.

The source-campaign fixture fits all 17 training and four validation examples
through the eleven binary heads, retains source ordering, serializes and
restores the resulting identity. It admits no test examples and performs no
candidate evaluation. The affected source fitting, relation, custody and
crossfit checks passed: 102 tests.

This proves the bounded execution path on those fixtures. It does not measure
full-cohort peak memory, runtime, semantic gain or convergence. The detached
acquisition is still running at this record. G03 remains open.

## Repository Verification

The checkpoint passed `make smoke lint compile governance-lint layering writing
doc-drift`: smoke reported 164 passed and one skipped; the remaining gates
passed their declared baselines. These gates do not add a semantic gain claim.
