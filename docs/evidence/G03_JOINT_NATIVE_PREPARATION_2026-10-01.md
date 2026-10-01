# Joint Native Preparation, 2026-10-01

The source-only joint fit now has a preparation mode. It freezes the same
plan used by fitting without loading backbone weights or acquiring the model
lane. Source identities, observed evidence, supervision, token sequences,
public spans, schedule, loss settings, native topology, implementation and
installed arithmetic are bound into that plan. A changed plan cannot reuse
the prepared custody directory.

Preparation validates the actual source labels, complete checkpoint schedule,
fit-only sampling and witnessed equivariance maps. It rejects a training-only
domain adversary with fewer than two source environments. Span bounds and
token limits are checked before loading weights. `MLX_ENABLE_TF32=0` must be
set at process launch; descriptor and arithmetic are rechecked before model
acquisition and after fitting.

Adapter kinds and ranks can differ across suffix layers. They use the existing
native topology-aware implementation, not assumed attention projections.
Preparation projects adapter, pointer and optional nuisance-head parameter
storage. It reports activation memory as unmeasured. The parameter estimate
does not promise that a long sequence fits the host envelope.

## Checks

The affected source, native, binding, bridge, adapter, sampler and state-store
suites passed 105 tests in 25.37 seconds. The native fixture covers preparation
followed by the exact fit, changed plans rejected without another model load,
mixed function classes and ranks, missing launch arithmetic, invalid schedules
and invalid adversarial supervision. Exact interrupted/uninterrupted fitting
and completion reconciliation remain covered.

Lint, compile, governance ownership and architectural layering passed without
relaxing baselines. This is preparation and implementation evidence, not
native language accuracy, transfer or G03 closure. No failed historical fit or
negative bank is reclassified by this build. The existing native v7 evaluator
does not accept this new joint artifact; its integrated decode path must be
qualified explicitly before an expensive fit and broader measurement.
