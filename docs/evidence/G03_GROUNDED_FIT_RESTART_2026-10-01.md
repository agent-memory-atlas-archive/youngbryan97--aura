# Grounded Fit Restart, 2026-10-01

The integrated binding build is on main at `a6d704c37`, followed by the
conflict-free main merge `0d31abe5b`. This checkpoint adds recoverable joint
native/pointer fitting. G03 remains open. No new held-language run or live
serving change is recorded here.

## One Complete Generation

`fit_grounded_binding` saves trainable pointer/native parameters, Adam step
and moments, the nuisance head, group weights, RNG state, checkpoint history
and the selected candidate in one content-addressed safetensors generation.
The mutable `resume.json` pointer moves only after that generation is written.
The fit identity binds source observations and supervision, implementation,
native topology, loss settings and the complete source sampling schedule.

`--resume` restores a complete generation. Changed source custody, learning
rate or implementation is rejected. Stable, bounded reads reject final-path
symlinks and checksum failures. A changed mutable selected file is restored
from the generation. A completed fit is not trained again.

The exact interrupted/uninterrupted test compares every final tensor and the
selected checkpoint. It covers the pointer with Adam, group reweighting and
the training-only adversary. A separate small Qwen2 test compares the actual
joint native suffix and pointer after interruption. Both comparisons require
bitwise equality, not a tolerance or equal aggregate accuracy. These are
small implementation fixtures, not the resident 27B or unseen-family proof.

## Native Ownership And Completion

The native wrapper reuses verified immutable prefix shards. Model objects
are released before the lane exits on success and failure; weak references
check this in the native fixture. Completed exception frames are cleared
because they can otherwise retain the model after a failed fit.

If training finishes but completion publication fails, continuation verifies
the saved source fit, source supervision, adapter inventory and every retained
prefix/span receipt. It publishes the missing completion without loading the
backbone or repeating any update. Unavailable historical peak-memory data is
recorded as null, not inferred from the current host or the parameter budget.

Initial native source states are observed once rather than twice. Their
receipts are retained as bounded metadata, not a second full activation bank.
Initial acquisition is inside the declared fit allowance and emits progress.

`tools/verify_semantic_grounded_fit.py` independently checks source partition
custody, terminal Adam count, history, selected checkpoint tensors, saved
generation and native completion in a separate process. It loads adapters
and pointer arrays, not the backbone. Exit zero requires a positive-step
selected checkpoint. It grants no semantic qualification or serving authority.

## Checks

The affected acquisition, binder, native fit, chart bridge, adapter, sampling
and frozen-state suites passed 98 tests in 24.07 seconds. The native restart
fixture passed its five cases in 7.64 seconds, including a missing completion
receipt and changed supervision refused before another model load.

The earlier repository-gate pass completed smoke (164 passed, one skipped),
lint, compile, governance lint, layering and writing. After the complete
recovery change, all those gates passed again. Smoke passed 164 tests with
one skip in 57.29 seconds. Governance ownership matched its existing baseline;
no ceiling or acceptance standard was relaxed.

## Qualification Boundary

This closes the optimizer-restart and completion-reconciliation builds. It
does not close G03, G04, general gain, fusion or frontier performance. Before
the next costly run, the joint artifact still needs exact source preparation,
detached custody and the target-blind integrated decode qualification path.
Historical completed source fitting and the negative bank are retained; they
are not restarted or relabeled as successful semantic evidence.
