# Shared causal-prefix computation

The source-hash-selected FP32 trie probe completed in 157.48571274999995
seconds. It checked all 56 alternatives across the source's 12 actual grammar
decisions. Every within-decision ranking agrees with full-sequence FP32
execution. The largest target log-probability error is
0.0000438690185546875, below the unchanged 0.015625 allowance.

Full computation took 105.34588920904207 seconds. Trie computation, including
anchor construction and the unchanged suffix, took 12.483673705995898 seconds.
The prefix executed 324 tokens instead of 7,912 logical tokens; 2,033
continuation tokens were reused. Peak retained distinct cache states was 15.
The trie holds depth-first read-only ancestors and gives each advancing child
its own deep copy. These are within-run measurements on one source, not a
campaign throughput estimate or a reasoning result.

Directory: `semantic-native-grammar-choice-trie-float32-probe-20260926`.
Plan: `98c31efdf2bcd893a7dd6c9d0b4c2098d957635506e1011b19b2f11098235663`.
Report: `0b2c866a7bad2828fe3656bb776b047abbb6cdd8458344e5ea2b964c0f9e848c`.

A separate CPU calculation verified the report digest, every recorded target
error, all 12 rankings, complete choice coverage, and token accounting from
the persisted arrays. It did not rerun the model. Native BF16 and this FP32
basis remain numerically different; old qualifications do not migrate.

## FP16 Rejection

The selective FP16 probe completed in 53.47483758299495 seconds. It retained
all eight native FP32 parameters and all 498 packed integer parameter objects,
and converted 1,349 BF16 parameters. Parameter storage stayed at
15,134,637,056 bytes. All three source groups failed the cache/full numerical
check despite preserving their rankings. Largest error: 0.04511833190917969.
Largest native/FP16 shift: 0.3436279296875.

Directory: `semantic-native-prefix-branch-float16-probe-v3-20260926`.
Plan: `a83c631e8576d5b09cd53b436c37cc3ee0da80ddf2aa0c329de5125bc09e04fb`.
Report: `339bc58a50e243368ae3227294cce8451e196973fb51a664adcfdb2037204ce1`.
Separate CPU accounting reproduced all three rejections from raw target arrays.
No FP16 training or serving path is authorized by this result.

## Training Connection

Training can opt into an explicit FP32 execution contract and source-grouped
trie capture. Both checkpoint-bank replay and target-blind grammar generation
restore the declared arithmetic after loading weights. They check the installed
MLX implementation and launch arithmetic against the training plan. Historical
plans retain their native full-sequence behavior.

The training capture policy is fixed before measurement: check every choice
and within-decision ranking of the first source, then the first choice of every
later source. All sequences are captured; the policy does not inspect outcomes
or construction labels. These sampled checks do not prove equivalence on every
unchecked continuation.

The eight-update FP32 trie canary completed in 627.1726179999969 seconds.
It captured all 519 sequences over 12 sources in 418.0536436669936 seconds.
Complete-model versus suffix logits agreed exactly on its first check.
Calibration selected checkpoint four: loss 0.2489486038684845, compared with
0.3050346318632364 unfitted and 0.6182868257164955 at update eight.
Independent CPU regrading verified all 108 supervised decisions and the
reported held-out totals: 2/4 native, 2/4 incumbent, 2/4 unfitted, no gains
and no regressions. Only eight of 303 fitting sources received updates.

Directory: `semantic-native-grammar-choice-fp32-trie-canary-v1-20260926`.
Plan: `b13a5c496590f72aed6a0a42849b334f9d5a37a6a3030c4239243d4b5faba331`.
Report: `5b60cd052c7922f009dfe8129ca4adb75688340568d46ae24da2e611406771c6`.
No promotion, fusion, broad gain, or G03 closure follows from this canary.

The idle desktop was quit through its supported native application action.
Logs recorded clean service teardown and a semaphore-cleanup warning. No
blanket process kill was used. The experiment requests its own model lease;
the desktop is not proof that this research mechanism serves live requests.

## Complete-Fit Preparation

Source-sharded storage retains every captured alternative in immutable
safetensors, preserving dtype, shape, and values without quantization. Reads
verify the bytes through the existing stable-file gateway before loading.
The retained shard cache has an explicit byte bound; this is not a bound on
the model, optimizer, temporary capture tensors, or MLX allocator caches.
CPU verification checks shard coverage against the supervised token sequences
and all durable file digests. It does not independently recompute hidden states.

Focused real-MLX tests establish bitwise value round trips, preserved dtypes,
bounded cache eviction, unchanged loss and gradients, and refusal of corruption
or missing alternatives. Training writes a durable receipt after every source
and now requires an exclusive lease without evicting another model owner.
These changes prepare a complete fitting epoch; no completed full-fit result
is claimed here.

The CPU-only supervision audit for
`semantic-native-grammar-choice-fp32-trie-epoch-shards-v1-20260926` records
303/303 fitting sources scheduled once, all 185 calibration sources, 50
held-out sources, and 21,733 complete sequences of length 97 through 207.
Decision coverage is 2,208 references, 1,104 operations, and 1,104 terminations.
No model weights were loaded for this inventory.
Plan: `2ccc7d5f23dae9f99827e56003479619c52bff23417c5b597335be3ece19fa6d`.
Supervision: `1915ff910dec59bfce7fd333e7c5c7f27c7e8a9114ca324edf2370f3f4c81da9`.
