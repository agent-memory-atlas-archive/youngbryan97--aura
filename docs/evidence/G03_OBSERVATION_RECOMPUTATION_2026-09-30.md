# G03 observation-driven recomputation, 2026-09-30

## Connected Observations

The [scoped computation loop](../SCOPED_COMPUTATIONAL_KNOWLEDGE.md) now asks
explicit observation ports for absent input identities. A port returns typed
premises with scope, provenance, kind and validity. Text, derived values,
unrequested identities, duplicates and stale observations are refused.
Disagreement between ports remains an unresolved input.

Every successful round adds an absent declared input and reruns the equation
graph. Completed intermediate values are recalculated from the inputs; they
are not admitted back as observations. No-progress ends the loop without a
timer. The v2 receipt binds the initial context, each observation round and
the resulting portfolio decision. Estimated inputs remain conditional.

The test removes inventory evidence, supplies it through an asynchronous
observation port, recalculates the balance and selects the compatible executed
program. A separate test supplies inputs one at a time. Another starts with a
partly completed dependency graph and proves that recomputation does not reuse
old derived values. Conflict and unavailable-observation tests preserve gaps.

An optional intentional-memory connection uses the existing retrieval router.
The integration test creates a real SQLite local corpus, registers its adapter,
queries it through that router and preserves reference provenance. Retrieved
text does not become a current measurement. No second memory store is created.

Ports remain caller-bound authorities. This build does not discover a correct
language-to-equation mapping, automatically activate itself in live chat, or
make an observation true merely because a provider returned it.

## Source Partition Preparation

`tools/freeze_semantic_source_folds.py` now has an explicit `--source-fit-only`
mode. It requires `--preparation-receipt`, verifies the fitted model and source
report, then reuses the existing bundle, representation and source-split checks.
It freezes training constructions without first running candidate evaluation.

The companion receipt binds the candidate, source report, feature manifests
and fold receipt. It states that preparation performs no candidate evaluation
and confers no qualification or serving authority. The default measured-report
path still requires its verified candidate report. Tampered parents and
incompatible bundles remain errors. Validation and test examples do not enter
fold assignments.

## Checks

The computation, portfolio, local-corpus, intentional-retrieval, source-fold,
source-custody and crossfit checks passed: 156 tests. The preceding observation
build passed smoke (164 passed, one skipped), lint, compile, governance lint,
layering, writing and doc drift. Gates after the source-partition change remain
to be recorded separately.

The detached nine-cohort acquisition is still running. Eight cohort manifests
are present; the termination cohort is incomplete at this observation. No new
neural fit, generated gain, promotion or G03 closure is claimed.

## Checkpoint Verification

The preparation path now also refuses missing, false or numeric completion
flags; a source fit must explicitly report completion. The affected preparation,
observation, source-custody and crossfit checks passed: 60 tests. Smoke passed
164 tests with one skip. Lint, compile, governance lint, layering, writing and
doc drift passed. The new evidence file must be in Git's inventory for the
documentation checker to resolve the master-ledger link; the staged inventory
has zero broken references.
