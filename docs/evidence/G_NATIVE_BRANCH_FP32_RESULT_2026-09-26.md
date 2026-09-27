# FP32 source-cache arithmetic

The precision-controlled resident probe completed in 142.76791437499924 seconds.
Full FP32 execution and cached FP32 execution pass on all twelve alternatives
across three sources. Complete rankings agree. Per-target log-probability errors
stay below the unchanged 0.015625 allowance; their largest value is
0.0000762939453125.

This locates a numerical boundary in the native BF16 cache rejection. The cache
ownership and complete-sequence path can reproduce the full decoder under this
FP32 execution basis. It does not prove that every cache boundary or request is
equivalent, and it does not rehabilitate the BF16 measurement.

The arithmetic change is itself not native-equivalent at the same allowance.
Although all three native/FP32 alternative rankings agree, individual target
log probabilities shift too far. Their maxima include 0.3403434753417969.
Training, replay, and qualification would need to bind the new arithmetic basis;
this experiment grants no transfer of the old model-basis qualification.

Packed integer parameter objects remain unchanged at 498 sites. Floating
parameter storage rises from 15,134,637,056 to 16,820,762,624 total parameter
bytes. Full FP32 work took 69.22692029002064 seconds; cached FP32 work took
40.01432504101831 seconds, including anchor construction. These are within-run
measurements. They establish no speed improvement over the ordinary BF16 lane.

Directory: `semantic-native-prefix-branch-float32-probe-v2-20260926`.
Plan: `e66b0fe99c8ae84b7f9e681168e6617a53c790f0b0a00c2e4e94e380f11eb711`.
Report: `539c08c4e6cc353950d97afeab92aa745f98676a66360f92efa4ae414babbf16`.

The exclusive process lease was released and the probe process exited. No
training, serving, or admission default changed. G03 remains open.
