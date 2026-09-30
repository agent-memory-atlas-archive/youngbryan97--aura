# G03 early-stop contrast, 2026-09-29

The first stop corpus paired complete three-step fork/join graphs with
four-step continuations. That gave a late stop/continue contrast, but the
retained native miss at `sequence-cataphoric-4` stopped after one operation.
A late contrast alone is not direct supervision for that first possible stop.
The miss is diagnostic context, not a label used to choose a fit source.

`counterfactual_fork_join_stop_source_v2` keeps the v1 stop corpus unchanged
and adds a one-step prefix for each of its 648 fit cells. Its three depths are
1, 3, and 4, with 648 source examples at each depth. Each one-step prefix
uses the same public inputs and first operation as its three-step partner,
then asks to return that first result. The builder requires a witnessed output
change; the typed teacher paths first differ at the stop/continue decision.
The witness may use a counterfactual input: 13 of the 648 pairs happen to
produce the same answer on their printed values. They remain different
programs, not evidence of a public-answer gain on those 13 examples.
All examples are training-only. The feature builder refuses any cap below the
entire 1,944-source cohort, because generic cell coverage is not enough to
retain both sides of every stop contrast.

The selected example-ID list has SHA-256
`9377280d28c8671f0df275af97d71b17a48e7e6e3dbaec3ea29f96704b5154fa`.
Using the active 27B tokenizer named in the
[`real-tokenizer audit`](G03_REAL_TOKENIZER_PAIR_AUDIT_2026-09-29.md),
without loading weights, all 1,944 source annotations projected. Maximum
source lengths were 78, 125, and 144 tokens at depths 1, 3, and 4. The
fit-only typed planner found witnessed operation, reference, and termination
peers for all 1,944 sources. Decision 3, the first stop, is paired in all
1,944; decision 11, the later stop, is paired in the 1,296 depth-three/four
sources. This is a CPU-only supervision inventory, not a measured gain.

The current resident-model micro campaign still owns the model lane. No v2
hidden states, training fit, native generation, baseline preservation,
held-family transfer, fusion, or serving result exists from this record.
G03 remains open.
