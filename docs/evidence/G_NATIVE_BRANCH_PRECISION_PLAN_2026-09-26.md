# Precision-controlled source-cache probe

The native BF16 source-cache probe failed the unchanged 0.015625 per-target
log-probability allowance on all three sources. Identical rankings did not
authorize a different numerical execution contract.

The new opt-in probe tests another arithmetic basis. It retains packed integer
quantized weights, casts floating parameters to FP32, and requires
`MLX_ENABLE_TF32=0` at process launch. MLX documents that FP32 matrix-family
operations may otherwise use reduced precision on supported hardware:
[MLX numerical precision](https://ml-explore.github.io/mlx/build/html/usage/precision.html).
The installed versions and hybrid kernel source hashes are pinned in the plan.

Before casting, the probe records full-sequence native scores for every retained
alternative. It then compares full FP32 execution with cached FP32 execution,
including every frozen layer and the original suffix computation. The absolute
per-target allowance stays 0.015625. The complete alternative ranking must also
agree. The native-to-FP32 comparison is separate: changed rankings or excessive
target errors prevent calling the arithmetic change native-equivalent.

This measures whether precision can resolve the rejected cache approximation.
It does not declare FP32 better, change training or serving defaults, relabel
the previous rejection, or authorize an optimization from a small probe.

The shared model controller grants an exclusive process lease. The probe
refuses to evict another owner. The desktop is shut down through its supported
lifecycle before this model load. A second decoder cannot be admitted beside
the probe through the controller.

Directory: `semantic-native-prefix-branch-float32-probe-v2-20260926`.
Plan: `e66b0fe99c8ae84b7f9e681168e6617a53c790f0b0a00c2e4e94e380f11eb711`.
Population: three source-hash selected requests, four alternatives each.
Bound: 1800 seconds. All 39 focused precision/cache tests pass.

An earlier plan-only directory was superseded by the final import formatting
and explicit exclusive-lease requirement. No model measurement occurred there.
The active plan above binds the final implementation. Completion and its
numerical verdict remain unmeasured at this record.
