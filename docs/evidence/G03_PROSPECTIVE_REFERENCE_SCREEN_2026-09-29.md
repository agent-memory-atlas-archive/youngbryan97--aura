# G03 prospective reference screen, 2026-09-29

`tools/run_semantic_native_reference_screen.py` runs the existing native
evaluator on the three frozen natural reference requests at a smaller,
predeclared search bound. It freezes fitted, base, and source-erasure commands
before decode. Each arm uses the same node and runtime allowance. Complete
reports are independently verified, then the existing three-arm comparator
adjudicates the result. A completed arm is reverified on continuation rather
than decoded again. A partial arm still requires a new declared attempt.

The default screen uses 16 nodes. The choice came from the
[retrospective budget frontier](G03_NATIVE_BUDGET_FRONTIER_2026-09-29.md),
which reconstructed all three fitted procedures from saved model scores at
that bound. The screen has not been run prospectively; it has no measured
wall-time saving yet.

The screen passes only if all three fitted procedures and public answers are
correct, neither procedure nor answer regresses against base, no fitted
answer is forced, and at least one gain changes under source erasure. A pass
authorizes the unchanged 256-node micro protocol as the next measurement. A
failure means this candidate is not worth that cost under the screen policy;
it does not prove the candidate would fail at 256 nodes. The screen grants no
G03/G04 closure, general-transfer claim, or serving authority.

The policy-only mode writes the nine frozen plan/decode/verify commands without
loading a model. It requires the same independently verified fit and source
calibration as the full micro path. An actual screen should run under the
detached supervisor when the model lane is free. This stage is for later
candidates; the source-matched 256-node campaign already running is unchanged.
