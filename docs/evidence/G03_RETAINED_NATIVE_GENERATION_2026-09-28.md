# Retained native generation, fitted versus base

The source-pair checkpoint selected at step 202 generated graphs without a
candidate bank on fourteen previously exposed validation requests, two per
source cohort. Both arms used the same public literal coordinates, native
typed grammar, eight-step bound, FP32 arithmetic, and exact frozen-prefix
trie. Neither scorer received the expected graph or answer.

| Measurement | Fitted | Base | Gains | Regressions |
| --- | ---: | ---: | ---: | ---: |
| Exact requested procedure | 7/14 | 1/14 | 7 | 1 |
| Answer on observed inputs | 9/14 | 3/14 | 7 | 1 |
| Proven output and domain equivalence | 8/14 | 2/14 | 7 | 1 |

Both arms were independently replayed by the CPU verifier with zero current
implementation drift. The new paired comparator checks exact agreement of
every plan field except weight mode and plan identity, then recomputes the
meaning audit from the source-bound graphs. It does not infer correctness
from model scores or process exit status.

The regression is `021b4921101b...`, a fork/join request. The fitted graph
binds the first subtraction to the wrong public input and yields 2793 rather
than 2325. The base graph does not preserve the requested procedure, but its
output and domain were proven equivalent by integer-polynomial normalization.
The fitted graph has a witnessed different meaning. This is not merely loss
of syntactic fidelity or uncertainty in the grader.

Another request, `005ccc967d8b...`, is procedure-inexact after adaptation but
remains meaning-equivalent: both additions commute. A matching answer on
`061caf3e1520...` remains unknown under the independent meaning comparison;
it is not promoted to a proof. The fitted arm has five witnessed meaning
failures and one unknown. The base has nine unavailable graphs, two witnessed
failures, and one unknown. Ten base requests reached the depth bound, versus
zero fitted requests. Completion and semantic selection are distinct gains.

The fitted run took 134.45 seconds; the base took 538.87 seconds. Their
generated paths and work differ, so this is not an equal-work speedup claim.
The first fitted launch failed admission before loading any model. Its
successful second attempt and the base run both have detached terminal
receipts with empty surviving process lineage.

Training already checks that supervised inputs equal runtime source-order
literal extraction. That suspected coordinate mismatch is not the cause.
The source-pair interaction objective has 261 operation contrasts and no
reference or termination contrasts, although ordinary conditional loss
supervises all three kinds. Average loss and candidate-bank ranking are not
equivalent to winning every choice needed for target-blind generation.

A complete-path source-calibration audit was built to locate that gap on
all 185 calibration sources and all four saved checkpoints. It counts the
first failed choice, complete teacher paths, and per-kind decisions; it
reuses immutable state shards and preserves the historical selection. Its
results are separate evidence, not presumed here.

Evidence under `~/.aura/rlc-evidence/`:

- `semantic-native-source-pairs-retained-trie-canary-v1-20260928`
- `semantic-native-source-pairs-retained-trie-base-canary-v1-20260928`
- `semantic-native-source-pairs-retained-trie-comparison-v1-20260928.json`

Comparison receipt:
`7abc8f52871dbc4e63a4a6fbda47c2883d58d161cf9d2a023c0bd97ed94b11c5`.
Training and checkpoint identities remain those of the paired-source fit.
This development canary is not promotable, fresh-family transfer, broad
reasoning gain, public-answer qualification, or fusion. G03 remains open;
the 500-request broad run is not launched from this result.
